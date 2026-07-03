"""Summary eval for DGRU/Transformer DPD at FL=50.

Two eval modes:
  (a) Paper-style: nperseg=2560 fixed segments → DGRU enjoys warmup dilution.
  (b) Sliding window FL=50 with stride=1 → fair architecture-only comparison.

PA = DGRU H=8 (frozen, paper-trained).
DPD: DGRU H=8 (486 params), Transformer H=6 (488 params, no-PE / with-PE).
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)

import models as M
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR


def get_target_gain(x_train, y_train):
    in_amp = np.abs(x_train[:,0] + 1j*x_train[:,1])
    out_amp = np.abs(y_train[:,0] + 1j*y_train[:,1])
    return float(np.max(out_amp) / np.max(in_amp))


def paper_eval(net_cas, X, target_full, nperseg, fs, bw_main, n_sub_ch, device):
    """Non-overlapping segments of length nperseg. Same as OpenDPD paper convention."""
    N = X.shape[0]
    n_seg = N // nperseg
    X_seg = X[:n_seg*nperseg].reshape(n_seg, nperseg, 2)
    tgt_seg = target_full[:n_seg*nperseg].reshape(n_seg, nperseg, 2)
    x_t = torch.from_numpy(X_seg.astype(np.float32)).to(device)
    net_cas.eval()
    with torch.no_grad():
        pred = net_cas(x_t).cpu().numpy()
    nmse = NMSE(pred, tgt_seg)
    evm = EVM(pred, tgt_seg, sample_rate=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub_ch, nperseg=nperseg)
    aclr_l, aclr_r = ACLR(pred, fs=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub_ch, nperseg=nperseg)
    return nmse, evm, aclr_l, aclr_r, n_seg


def sliding_eval(net_cas, X, target_full, FL, nperseg, fs, bw_main, n_sub_ch, device, batch=256):
    """FL-window sliding stride=1 → take last position. Then chunk into nperseg for FFT metrics."""
    N = X.shape[0]
    M_out = N - FL + 1
    out = np.zeros((M_out, 2), dtype=np.float32)
    x_t = torch.from_numpy(X.astype(np.float32))
    net_cas.eval()
    with torch.no_grad():
        for start in range(0, M_out, batch):
            end = min(start+batch, M_out)
            idx = torch.arange(start, end).unsqueeze(1) + torch.arange(FL).unsqueeze(0)
            win = x_t[idx].to(device)
            cas = net_cas(win)
            out[start:end] = cas[:, -1, :].cpu().numpy()
    target_aligned = target_full[FL-1:]
    n_seg = out.shape[0] // nperseg
    pred_seg = out[:n_seg*nperseg].reshape(n_seg, nperseg, 2)
    tgt_seg = target_aligned[:n_seg*nperseg].reshape(n_seg, nperseg, 2)
    nmse = NMSE(pred_seg, tgt_seg)
    evm = EVM(pred_seg, tgt_seg, sample_rate=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub_ch, nperseg=nperseg)
    aclr_l, aclr_r = ACLR(pred_seg, fs=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub_ch, nperseg=nperseg)
    return nmse, evm, aclr_l, aclr_r, n_seg


def main():
    device = torch.device('cuda:0')
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name='DPA_200MHz')
    G = get_target_gain(X_train, y_train)
    print(f"target_gain = {G:.4f}")

    with open('datasets/DPA_200MHz/spec.json') as f: spec = json.load(f)
    fs = float(spec['input_signal_fs']); bw_main = float(spec['bw_main_ch']); n_sub_ch = int(spec['n_sub_ch']); nperseg = int(spec['nperseg'])
    print(f"fs={fs/1e6:.1f}MHz, bw_main={bw_main/1e6:.0f}MHz, nperseg={nperseg}")

    target_test = G * X_test

    pa_ckpt = 'save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt'
    runs = [
        ('DGRU H=8',         'dgru',        8, {},  'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_50/DPD_S_0_M_DGRU_H_8_F_50_P_486.pt'),
        ('TRANSFORMER no-PE','transformer', 6, dict(n_heads=2, d_ff=18, use_pos_encoding=False), 'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_50/DPD_S_0_M_TRANSFORMER_H_6_F_50_P_488_PE_0.pt'),
        ('TRANSFORMER w/PE', 'transformer', 6, dict(n_heads=2, d_ff=18, use_pos_encoding=True),  'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_50/DPD_S_0_M_TRANSFORMER_H_6_F_50_P_488_PE_1.pt'),
    ]

    rows = []
    for tag, bb_type, h_size, extra, dpd_ckpt in runs:
        if not os.path.exists(dpd_ckpt):
            print(f"SKIP {tag}: ckpt missing"); continue
        net_pa  = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
        net_pa.load_state_dict(torch.load(pa_ckpt, map_location='cpu')); net_pa.eval()
        net_dpd = M.CoreModel(input_size=2, hidden_size=h_size, num_layers=1, backbone_type=bb_type, **extra)
        sd = torch.load(dpd_ckpt, map_location='cpu')
        sd = {k: v for k, v in sd.items() if not k.endswith('pos_encoding.pe')}
        net_dpd.load_state_dict(sd, strict=False)
        net_dpd.eval()
        net_cas = M.CascadedModel(dpd_model=net_dpd, pa_model=net_pa).to(device).eval()

        # (a) Paper-style eval (nperseg=2560 fixed)
        a = paper_eval(net_cas, X_test, target_test, nperseg, fs, bw_main, n_sub_ch, device)
        # (b) Sliding window FL=50
        b = sliding_eval(net_cas, X_test, target_test, FL=50, nperseg=nperseg, fs=fs, bw_main=bw_main, n_sub_ch=n_sub_ch, device=device)

        rows.append({'model': tag, 'eval': 'paper(nperseg=2560)',  'NMSE': a[0], 'EVM': a[1], 'ACLR_L': a[2], 'ACLR_R': a[3], 'ACLR_AVG': (a[2]+a[3])/2, 'n_seg': a[4]})
        rows.append({'model': tag, 'eval': 'sliding-window(FL=50)', 'NMSE': b[0], 'EVM': b[1], 'ACLR_L': b[2], 'ACLR_R': b[3], 'ACLR_AVG': (b[2]+b[3])/2, 'n_seg': b[4]})
        print(f"  {tag} | paper:   NMSE={a[0]:7.2f} EVM={a[1]:7.2f} ACLR_AVG={(a[2]+a[3])/2:7.2f}  n_seg={a[4]}")
        print(f"  {tag} | sliding: NMSE={b[0]:7.2f} EVM={b[1]:7.2f} ACLR_AVG={(b[2]+b[3])/2:7.2f}  n_seg={b[4]}")

    df = pd.DataFrame(rows)
    df.to_csv('bash_scripts/fl50_summary.csv', index=False, float_format='%.3f')
    print("\nSaved bash_scripts/fl50_summary.csv\n")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
