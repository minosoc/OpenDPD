"""rev4: Sliding eval for Transformer no-PE variants at FL=50.

V0: baseline (488p)
V1: + output_residual_concat (500p)
V2: + input/output MLP h=12 (704p)
DGRU H=8 reference (486p)
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


def sliding_pred(net_cas, x, FL, device, batch=256):
    x_t = torch.from_numpy(x.astype(np.float32))
    N = x.shape[0]; M_out = N - FL + 1
    out = np.zeros((M_out, 2), dtype=np.float32)
    net_cas.eval()
    with torch.no_grad():
        for s in range(0, M_out, batch):
            e = min(s+batch, M_out)
            idx = torch.arange(s, e).unsqueeze(1) + torch.arange(FL).unsqueeze(0)
            win = x_t[idx].to(device)
            out[s:e] = net_cas(win)[:, -1, :].cpu().numpy()
    return out


def main():
    device = torch.device('cuda:0')
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name='DPA_200MHz')
    G = float(np.max(np.abs(y_train[:,0]+1j*y_train[:,1])) / np.max(np.abs(X_train[:,0]+1j*X_train[:,1])))
    target_test = G * X_test
    with open('datasets/DPA_200MHz/spec.json') as f: spec = json.load(f)
    fs = float(spec['input_signal_fs']); bw_main = float(spec['bw_main_ch']); n_sub = int(spec['n_sub_ch'])
    nperseg = int(spec['nperseg'])
    print(f"fs={fs/1e6:.1f}MHz bw_main={bw_main/1e6:.0f}MHz nperseg={nperseg} G={G:.4f}")

    pa_ckpt = 'save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt'
    FL = 50

    runs = [
        # (tag, backbone, h_size, extras, ckpt_path)
        ('DGRU H=8 (486p)',                  'dgru',        8, {},
         'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_50/DPD_S_0_M_DGRU_H_8_F_50_P_486.pt'),
        ('V0 baseline (488p)',               'transformer', 6, dict(n_heads=2, d_ff=18, use_pos_encoding=False),
         'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_50/DPD_S_0_M_TRANSFORMER_H_6_F_50_P_488_PE_0.pt'),
        ('V1 +residual concat (500p)',       'transformer', 6, dict(n_heads=2, d_ff=18, use_pos_encoding=False, output_residual_concat=True),
         'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_50/DPD_S_0_M_TRANSFORMER_H_6_F_50_P_500_PE_0.pt'),
        ('V2 +I/O MLP h=12 (704p)',          'transformer', 6, dict(n_heads=2, d_ff=18, use_pos_encoding=False, input_mlp_hidden=12, output_mlp_hidden=12),
         'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_50/DPD_S_0_M_TRANSFORMER_H_6_F_50_P_704_PE_0.pt'),
    ]
    rows = []
    for tag, bb_type, hs, extras, dpd_ckpt in runs:
        if not os.path.exists(dpd_ckpt):
            print(f"SKIP {tag}: missing ckpt")
            continue
        net_pa = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
        net_pa.load_state_dict(torch.load(pa_ckpt, map_location='cpu')); net_pa.eval()
        net_dpd = M.CoreModel(input_size=2, hidden_size=hs, num_layers=1, backbone_type=bb_type, **extras)
        sd = torch.load(dpd_ckpt, map_location='cpu')
        sd = {k: v for k, v in sd.items() if not k.endswith('pos_encoding.pe')}
        net_dpd.load_state_dict(sd, strict=False); net_dpd.eval()
        net_cas = M.CascadedModel(dpd_model=net_dpd, pa_model=net_pa).to(device).eval()

        pred = sliding_pred(net_cas, X_test, FL, device)
        tgt = target_test[FL-1:]
        n_seg = pred.shape[0] // nperseg
        pred_seg = pred[:n_seg*nperseg].reshape(n_seg, nperseg, 2)
        tgt_seg = tgt[:n_seg*nperseg].reshape(n_seg, nperseg, 2)
        nmse = NMSE(pred_seg, tgt_seg)
        evm = EVM(pred_seg, tgt_seg, sample_rate=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub, nperseg=nperseg)
        aclr_l, aclr_r = ACLR(pred_seg, fs=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub, nperseg=nperseg)
        aclr_avg = (aclr_l + aclr_r) / 2
        n_params = sum(p.numel() for p in net_dpd.parameters())
        print(f"  {tag:35s} | DPD_params={n_params:5d} | NMSE={nmse:7.2f} EVM={evm:7.2f} ACLR_L={aclr_l:7.2f} ACLR_R={aclr_r:7.2f} ACLR_AVG={aclr_avg:7.2f}")
        rows.append({'model': tag, 'dpd_params': n_params, 'NMSE': nmse, 'EVM': evm,
                     'ACLR_L': aclr_l, 'ACLR_R': aclr_r, 'ACLR_AVG': aclr_avg, 'n_seg': n_seg})
    df = pd.DataFrame(rows)
    os.makedirs('results/rev4_attn_variants', exist_ok=True)
    df.to_csv('results/rev4_attn_variants/rev4_attn_variants_results.csv', index=False, float_format='%.3f')
    print("\nSaved results/rev4_attn_variants/rev4_attn_variants_results.csv")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
