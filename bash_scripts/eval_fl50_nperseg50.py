"""Evaluate FL=50 paper-trained models with nperseg=50 (segments matching train length).

This treats the test set as non-overlapping 50-sample segments — same length as
the training frame. Drops any partial trailing segment to avoid zero-padding bias.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)

import models as M
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR


def get_target_gain(x_train, y_train):
    in_amp = np.abs(x_train[:, 0] + 1j * x_train[:, 1])
    out_amp = np.abs(y_train[:, 0] + 1j * y_train[:, 1])
    return float(np.max(out_amp) / np.max(in_amp))


def eval_nperseg(net_cas, X, target_full, nperseg, fs, bw_main, n_sub_ch, device):
    """Non-overlapping segments of length nperseg (drop trailing partial)."""
    N = X.shape[0]
    n_seg = N // nperseg
    X_seg = X[:n_seg * nperseg].reshape(n_seg, nperseg, 2)
    tgt_seg = target_full[:n_seg * nperseg].reshape(n_seg, nperseg, 2)
    x_t = torch.from_numpy(X_seg.astype(np.float32)).to(device)
    net_cas.eval()
    with torch.no_grad():
        pred = net_cas(x_t).cpu().numpy()
    nmse = NMSE(pred, tgt_seg)
    evm = EVM(pred, tgt_seg, sample_rate=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub_ch, nperseg=nperseg)
    aclr_l, aclr_r = ACLR(pred, fs=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub_ch, nperseg=nperseg)
    return nmse, evm, aclr_l, aclr_r, n_seg


def main():
    device = torch.device('cuda:0')
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name='DPA_200MHz')
    G = get_target_gain(X_train, y_train)
    with open('datasets/DPA_200MHz/spec.json') as f: spec = json.load(f)
    fs = float(spec['input_signal_fs']); bw_main = float(spec['bw_main_ch']); n_sub_ch = int(spec['n_sub_ch'])
    print(f"fs={fs/1e6:.1f}MHz, bw_main={bw_main/1e6:.0f}MHz, n_sub_ch={n_sub_ch}, target_gain={G:.4f}")

    target_test = G * X_test
    pa_ckpt = 'save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt'

    runs = [
        ('DGRU H=8',          'dgru',        8, {}, 'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_50/DPD_S_0_M_DGRU_H_8_F_50_P_486.pt'),
        ('Transformer no-PE', 'transformer', 6, dict(n_heads=2, d_ff=18, use_pos_encoding=False), 'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_50/DPD_S_0_M_TRANSFORMER_H_6_F_50_P_488_PE_0.pt'),
        ('Transformer w/PE',  'transformer', 6, dict(n_heads=2, d_ff=18, use_pos_encoding=True),  'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_50/DPD_S_0_M_TRANSFORMER_H_6_F_50_P_488_PE_1.pt'),
    ]
    nperseg_list = [50, 2560]
    rows = []
    for tag, bb_type, h_size, extra, dpd_ckpt in runs:
        net_pa = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
        net_pa.load_state_dict(torch.load(pa_ckpt, map_location='cpu')); net_pa.eval()
        net_dpd = M.CoreModel(input_size=2, hidden_size=h_size, num_layers=1, backbone_type=bb_type, **extra)
        sd = torch.load(dpd_ckpt, map_location='cpu')
        sd = {k: v for k, v in sd.items() if not k.endswith('pos_encoding.pe')}
        net_dpd.load_state_dict(sd, strict=False); net_dpd.eval()
        net_cas = M.CascadedModel(dpd_model=net_dpd, pa_model=net_pa).to(device).eval()
        for nps in nperseg_list:
            res = eval_nperseg(net_cas, X_test, target_test, nps, fs, bw_main, n_sub_ch, device)
            nmse, evm, aclr_l, aclr_r, n_seg = res
            rows.append({
                'model': tag, 'nperseg': nps,
                'NMSE': nmse, 'EVM': evm,
                'ACLR_L': aclr_l, 'ACLR_R': aclr_r, 'ACLR_AVG': (aclr_l+aclr_r)/2,
                'n_seg': n_seg
            })
            print(f"  {tag:22s} | nperseg={nps:5d} | NMSE={nmse:7.2f} EVM={evm:7.2f} ACLR_L={aclr_l:7.2f} ACLR_R={aclr_r:7.2f} ACLR_AVG={(aclr_l+aclr_r)/2:7.2f}  n_seg={n_seg}")
    df = pd.DataFrame(rows)
    df.to_csv('bash_scripts/fl50_nperseg_summary.csv', index=False, float_format='%.3f')
    print("\nSaved bash_scripts/fl50_nperseg_summary.csv")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
