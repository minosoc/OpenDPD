"""PSD comparison: input / PA only / DGRU cascade / Transformer no-PE cascade / Transformer w/PE cascade.

Shows that Transformer no-PE has good ACLR (out-of-band clean) but its in-band PSD
differs from target (explains EVM=-21 vs DGRU EVM=-43 with similar ACLR).
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.signal import welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)

import models as M
from modules.data_collector import load_dataset


def get_target_gain(x_train, y_train):
    in_amp = np.abs(x_train[:,0] + 1j*x_train[:,1])
    out_amp = np.abs(y_train[:,0] + 1j*y_train[:,1])
    return float(np.max(out_amp) / np.max(in_amp))


def cascade_full(net_cas, X, device):
    """Single forward pass on full test sequence."""
    net_cas.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(0).to(device)  # (1, N, 2)
        cas = net_cas(x_t).squeeze(0).cpu().numpy()  # (N, 2)
    return cas


def pa_only(net_pa, X, device):
    net_pa.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(0).to(device)
        y = net_pa(x_t).squeeze(0).cpu().numpy()
    return y


def psd_db(iq, fs, nperseg):
    z = iq[:,0] + 1j*iq[:,1]
    f, p = welch(z, fs=fs, nperseg=nperseg, return_onesided=False, scaling='spectrum')
    f = np.fft.fftshift(f); p = np.fft.fftshift(p)
    p_db = 10*np.log10(p / np.max(p) + 1e-30)
    return f/1e6, p_db


def main():
    device = torch.device('cuda:0')
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name='DPA_200MHz')
    G = get_target_gain(X_train, y_train)
    with open('datasets/DPA_200MHz/spec.json') as f: spec = json.load(f)
    fs = float(spec['input_signal_fs']); bw_main = float(spec['bw_main_ch'])

    target_test = G * X_test

    pa_ckpt = 'save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt'
    net_pa_only = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
    net_pa_only.load_state_dict(torch.load(pa_ckpt, map_location='cpu'))
    net_pa_only.eval().to(device)

    # Build 3 cascade models
    cascades = {}
    for tag, bb_type, h_size, extra, dpd_ckpt in [
        ('DGRU H=8', 'dgru', 8, {}, 'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_50/DPD_S_0_M_DGRU_H_8_F_50_P_486.pt'),
        ('Transformer no-PE', 'transformer', 6, dict(n_heads=2, d_ff=18, use_pos_encoding=False), 'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_50/DPD_S_0_M_TRANSFORMER_H_6_F_50_P_488_PE_0.pt'),
        ('Transformer w/PE', 'transformer', 6, dict(n_heads=2, d_ff=18, use_pos_encoding=True),  'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_50/DPD_S_0_M_TRANSFORMER_H_6_F_50_P_488_PE_1.pt'),
    ]:
        net_pa = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
        net_pa.load_state_dict(torch.load(pa_ckpt, map_location='cpu'))
        net_pa.eval()
        net_dpd = M.CoreModel(input_size=2, hidden_size=h_size, num_layers=1, backbone_type=bb_type, **extra)
        sd = torch.load(dpd_ckpt, map_location='cpu')
        sd = {k: v for k, v in sd.items() if not k.endswith('pos_encoding.pe')}
        net_dpd.load_state_dict(sd, strict=False); net_dpd.eval()
        cascades[tag] = M.CascadedModel(dpd_model=net_dpd, pa_model=net_pa).to(device).eval()

    # Compute outputs
    nperseg = 2560
    y_target = target_test  # G·X
    y_pa = pa_only(net_pa_only, X_test, device)
    y_dgru = cascade_full(cascades['DGRU H=8'], X_test, device)
    y_attn = cascade_full(cascades['Transformer no-PE'], X_test, device)
    y_attn_pe = cascade_full(cascades['Transformer w/PE'], X_test, device)

    # PSDs
    f_in, p_target = psd_db(y_target, fs, nperseg)
    _, p_pa = psd_db(y_pa, fs, nperseg)
    _, p_dgru = psd_db(y_dgru, fs, nperseg)
    _, p_attn = psd_db(y_attn, fs, nperseg)
    _, p_attn_pe = psd_db(y_attn_pe, fs, nperseg)

    # === Figure 1: All five overlaid, full range ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    ax.plot(f_in, p_target, '-', color='black', lw=1.5, label='target (G·X)')
    ax.plot(f_in, p_pa, '-', color='tab:red', lw=1.2, alpha=0.8, label='PA only (no DPD)')
    ax.plot(f_in, p_dgru, '-', color='tab:blue', lw=1.2, label='DGRU cascade')
    ax.plot(f_in, p_attn, '--', color='tab:orange', lw=1.4, label='Transformer no-PE cascade')
    ax.plot(f_in, p_attn_pe, ':', color='tab:green', lw=1.4, label='Transformer w/PE cascade')
    ax.axvspan(-bw_main/2e6, bw_main/2e6, color='gray', alpha=0.10, label='main ch (±100 MHz)')
    ax.set_xlim(-400, 400); ax.set_ylim(-80, 5)
    ax.set_xlabel('Frequency (MHz)'); ax.set_ylabel('Normalized PSD (dB)')
    ax.set_title('Full spectrum (out-of-band visible)')
    ax.grid(True, alpha=0.3); ax.legend(loc='lower center', fontsize=9, ncol=2)

    # === Figure 2: Zoom into main channel ===
    ax = axes[1]
    ax.plot(f_in, p_target, '-', color='black', lw=2, label='target (G·X)')
    ax.plot(f_in, p_dgru, '-', color='tab:blue', lw=1.4, label='DGRU cascade')
    ax.plot(f_in, p_attn, '--', color='tab:orange', lw=1.4, label='Transformer no-PE cascade')
    ax.plot(f_in, p_attn_pe, ':', color='tab:green', lw=1.4, label='Transformer w/PE cascade')
    ax.axvspan(-bw_main/2e6, bw_main/2e6, color='gray', alpha=0.10)
    ax.set_xlim(-120, 120); ax.set_ylim(-40, 5)
    ax.set_xlabel('Frequency (MHz)'); ax.set_ylabel('Normalized PSD (dB)')
    ax.set_title('Zoom: within main channel (in-band fidelity)')
    ax.grid(True, alpha=0.3); ax.legend(loc='lower center', fontsize=9)

    fig.suptitle('DPA_200MHz FL=50 — PSD comparison (paper config, ~486 params)', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig('bash_scripts/psd_fl50_comparison.png', dpi=140, bbox_inches='tight')
    print("Saved bash_scripts/psd_fl50_comparison.png")

    # === Figure 3: Difference from target (only in main band) ===
    fig2, ax = plt.subplots(1, 1, figsize=(10, 5))
    mask = np.abs(f_in) <= bw_main/2e6  # in main channel
    ax.plot(f_in[mask], (p_dgru - p_target)[mask], '-', color='tab:blue', lw=1.4, label='DGRU - target')
    ax.plot(f_in[mask], (p_attn - p_target)[mask], '--', color='tab:orange', lw=1.4, label='Transformer no-PE - target')
    ax.plot(f_in[mask], (p_attn_pe - p_target)[mask], ':', color='tab:green', lw=1.4, label='Transformer w/PE - target')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Frequency (MHz)'); ax.set_ylabel('PSD difference (dB)')
    ax.set_title('Cascade vs Target — in-band PSD difference\n(closer to 0 = better in-band fidelity → lower EVM)')
    ax.set_ylim(-10, 10)
    ax.grid(True, alpha=0.3); ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('bash_scripts/psd_fl50_inband_diff.png', dpi=140, bbox_inches='tight')
    print("Saved bash_scripts/psd_fl50_inband_diff.png")


if __name__ == '__main__':
    main()
