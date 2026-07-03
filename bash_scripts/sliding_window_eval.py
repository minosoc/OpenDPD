"""Sliding-window evaluation for sequence-length sweep.

For each (backbone, FL) checkpoint:
  - Load PA (DGRU H=8, fixed) and DPD (DGRU/Transformer at given FL)
  - Slide a FL-length window with stride=1 across X_test (and X_val)
  - At each window, take cascade output at the LAST position (output[:, -1, :])
  - Concatenate → full-length predicted stream (N - FL + 1 samples)
  - Compute NMSE / EVM / ACLR on this long stream vs target = G·X_test

This evaluates the model's "warmed-up" prediction quality, removing the
segment-warm-up bias that hurts RNNs at small FL.

Saves results to bash_scripts/sliding_window_results.csv.
"""
import os
import sys
import glob
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch

# Repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import models as M
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR


def get_target_gain(x_train, y_train):
    """Same as project.set_target_gain"""
    in_amp = np.abs(x_train[:, 0] + 1j * x_train[:, 1])
    out_amp = np.abs(y_train[:, 0] + 1j * y_train[:, 1])
    max_in_amp = np.max(in_amp)
    max_out_amp = np.max(out_amp)
    return float(max_out_amp / max_in_amp)


def sliding_cascade_output(net_cas, x, frame_length, device, batch=512):
    """Apply sliding window of length frame_length over x with stride=1.
    Take last-position output for each window.

    If frame_length > len(x), fall back to single full-sequence forward,
    taking all positions of the output (length = len(x)).

    Args:
        x: (N, 2) full sequence
        frame_length: FL
    Returns:
        Predicted output, length = max(N - FL + 1, N).
    """
    N = x.shape[0]
    if frame_length > N:
        # FL > N: can't slide. Single forward pass on whole sequence.
        net_cas.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)  # (1, N, 2)
            cas = net_cas(x_t)  # (1, N, 2)
        return cas[0].cpu().numpy()
    M_out = N - frame_length + 1
    out = np.zeros((M_out, 2), dtype=np.float32)

    # Build windows lazily in batches to limit RAM
    x_t = torch.from_numpy(x.astype(np.float32))
    net_cas.eval()
    with torch.no_grad():
        for start in range(0, M_out, batch):
            end = min(start + batch, M_out)
            B = end - start
            # Build (B, FL, 2) window tensor
            idx = torch.arange(start, end).unsqueeze(1) + torch.arange(frame_length).unsqueeze(0)
            win = x_t[idx]  # (B, FL, 2)
            win = win.to(device)
            cas = net_cas(win)  # (B, FL, 2)
            # Last position
            last = cas[:, -1, :].cpu().numpy()
            out[start:end] = last
    return out


def find_ckpt(save_dir, prefix):
    """Find first matching checkpoint."""
    pat = os.path.join(save_dir, prefix + '*.pt')
    files = sorted(glob.glob(pat))
    if not files:
        return None
    return files[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_name', default='DPA_200MHz')
    ap.add_argument('--device', default='cuda:5')
    ap.add_argument('--fls', default='50,100,500,1000,2000,3000,4000,5000,10000')
    ap.add_argument('--out_csv', default='bash_scripts/sliding_window_results.csv')
    ap.add_argument('--batch', default=512, type=int)
    args = ap.parse_args()

    device = torch.device(args.device)
    fl_list = [int(f) for f in args.fls.split(',')]

    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name=args.dataset_name)
    target_gain = get_target_gain(X_train, y_train)
    print(f"target_gain = {target_gain:.4f}")

    # Read spec for metrics
    with open(f'datasets/{args.dataset_name}/spec.json') as f:
        spec = json.load(f)
    fs = float(spec['input_signal_fs'])
    bw_main = float(spec['bw_main_ch'])
    n_sub_ch = int(spec['n_sub_ch'])
    nperseg_metric = int(spec.get('nperseg', 2560))
    print(f"spec: fs={fs/1e6:.1f}MHz, bw_main={bw_main/1e6:.1f}MHz, n_sub_ch={n_sub_ch}, nperseg={nperseg_metric}")

    # Cascade PA = DGRU H=8 (fixed across all)
    pa_dir = f'save/{args.dataset_name}/train_pa'
    pa_ckpt = os.path.join(pa_dir, 'PA_S_0_M_DGRU_H_8_F_50_P_486.pt')
    if not os.path.exists(pa_ckpt):
        print(f"ERROR: PA checkpoint not found: {pa_ckpt}")
        sys.exit(1)

    rows = []
    for FL in fl_list:
        for bb_tag, bb_type, h_size, extra_kw in [
            ('DGRU',        'dgru',        8, {}),
            ('ATTN(no PE)', 'transformer', 6, {'n_heads': 2, 'd_ff': 18, 'use_pos_encoding': False}),
            ('ATTN(PE)',    'transformer', 6, {'n_heads': 2, 'd_ff': 18, 'use_pos_encoding': True}),
        ]:
            # Find DPD checkpoint
            dpd_dir = f'save/{args.dataset_name}/train_dpd/PA_S_0_M_DGRU_H_8_F_{FL}'
            if bb_type == 'dgru':
                dpd_pat = f'DPD_S_0_M_DGRU_H_8_F_{FL}_P_'
            else:
                pe_flag = '1' if extra_kw['use_pos_encoding'] else '0'
                dpd_pat = f'DPD_S_0_M_TRANSFORMER_H_6_F_{FL}_P_488_PE_{pe_flag}'
            dpd_ckpt = find_ckpt(dpd_dir, dpd_pat)
            if dpd_ckpt is None:
                print(f"  SKIP {bb_tag} FL={FL} (no ckpt yet: {dpd_dir}/{dpd_pat}*)")
                continue
            print(f"\n=== {bb_tag} FL={FL} ===")
            print(f"  PA  : {pa_ckpt}")
            print(f"  DPD : {dpd_ckpt}")

            # Build PA + DPD
            net_pa = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
            net_pa.load_state_dict(torch.load(pa_ckpt, map_location='cpu'))
            net_pa.eval()
            net_dpd = M.CoreModel(input_size=2, hidden_size=h_size, num_layers=1, backbone_type=bb_type, **extra_kw)
            sd = torch.load(dpd_ckpt, map_location='cpu')
            # PE buffer may have grown during training to >max_len in checkpoint.
            # Drop the pe buffer; it will be regenerated lazily in forward.
            sd = {k: v for k, v in sd.items() if not k.endswith('pos_encoding.pe')}
            net_dpd.load_state_dict(sd, strict=False)
            net_dpd.eval()
            net_cas = M.CascadedModel(dpd_model=net_dpd, pa_model=net_pa).to(device).eval()

            # Sliding-window eval on X_test
            t0 = time.time()
            try:
                pred_test = sliding_cascade_output(net_cas, X_test, FL, device, batch=args.batch)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at FL={FL}, reducing batch")
                torch.cuda.empty_cache()
                pred_test = sliding_cascade_output(net_cas, X_test, FL, device, batch=max(args.batch // 16, 4))
            dt = time.time() - t0

            # Target = G·X_test, length-matched to predictions
            if FL > X_test.shape[0]:
                # Single full-sequence forward; pred has length N, align with full X_test
                target_test = target_gain * X_test
            else:
                target_test = target_gain * X_test[FL - 1:]  # align with sliding outputs
            assert pred_test.shape == target_test.shape, f"{pred_test.shape} vs {target_test.shape}"

            # Metrics: chunk into nperseg segments for FFT-based EVM/ACLR
            N = pred_test.shape[0]
            n_seg = N // nperseg_metric
            if n_seg < 1:
                print(f"  WARN: pred_test length {N} < nperseg {nperseg_metric}, skipping FFT metrics")
                continue
            pred_seg = pred_test[:n_seg * nperseg_metric].reshape(n_seg, nperseg_metric, 2)
            tgt_seg = target_test[:n_seg * nperseg_metric].reshape(n_seg, nperseg_metric, 2)

            nmse_db = NMSE(pred_seg, tgt_seg)
            evm_db = EVM(pred_seg, tgt_seg, sample_rate=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub_ch, nperseg=nperseg_metric)
            aclr_l, aclr_r = ACLR(pred_seg, fs=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub_ch, nperseg=nperseg_metric)
            aclr_avg = (aclr_l + aclr_r) / 2

            print(f"  NMSE = {nmse_db:7.2f} dB,  EVM = {evm_db:7.2f} dB,  ACLR_L = {aclr_l:7.2f} dB,  ACLR_R = {aclr_r:7.2f} dB,  ACLR_AVG = {aclr_avg:7.2f} dB  (n_seg={n_seg}, eval_time={dt:.1f}s)")

            rows.append({
                'backbone': bb_tag,
                'frame_length': FL,
                'NMSE': nmse_db,
                'EVM': evm_db,
                'ACLR_L': aclr_l,
                'ACLR_R': aclr_r,
                'ACLR_AVG': aclr_avg,
                'n_seg': n_seg,
                'eval_time_s': dt,
            })

    # Save CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.out_csv, index=False, float_format='%.4f')
        print(f"\nSaved {len(rows)} rows to {args.out_csv}")
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
