"""rev10e eval: k ∈ {21, 41} × lr ∈ {1e-3, 3e-4, 1e-4}.
Reads tagged ckpts (with _LR_<lr> suffix) for non-default lr.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import models as M
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR

FL = 200


def sliding_pred(net_cas, x, FL, device, batch=128):
    N = x.shape[0]; M_out = N - FL + 1
    out = np.zeros((M_out, 2), dtype=np.float32)
    x_t = torch.from_numpy(x.astype(np.float32))
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
    pa_ckpt = 'save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt'

    # rev10d default lr=1e-3 baselines — read from rev10d CSV (unlabeled ckpts were overwritten)
    rev10d_csv = pd.read_csv('results/rev10d_gmp_kernel_sweep/rev10d_results.csv')

    base = f'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_{FL}/'
    runs = []
    for k in [21, 41]:
        for lr in ['5e-4', '3e-4', '1e-4']:
            pat = base + f'DPD_S_0_M_TRANSFORMER_H_6_F_{FL}_*_LTL_1_PE_0_GMP_1_GSK_{k}_LR_{lr}.pt'
            cands = sorted(glob.glob(pat))
            if not cands:
                print(f"SKIP k={k} lr={lr}: no ckpt at {pat}"); continue
            runs.append((k, lr, cands[0]))

    rows = []
    for k, lr, ckpt in runs:
        net_pa = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
        net_pa.load_state_dict(torch.load(pa_ckpt, map_location='cpu')); net_pa.eval()
        net_dpd = M.CoreModel(input_size=2, hidden_size=6, num_layers=1, backbone_type='transformer',
                              n_heads=2, d_ff=10, use_pos_encoding=False,
                              use_gmp_stem=True, gmp_stem_kernel=k)
        net_dpd.load_state_dict(torch.load(ckpt, map_location='cpu')); net_dpd.eval()
        net_cas = M.CascadedModel(dpd_model=net_dpd, pa_model=net_pa).to(device).eval()
        pred = sliding_pred(net_cas, X_test, FL, device)
        tgt = target_test[FL-1:]
        n_seg = pred.shape[0] // nperseg
        pred_seg = pred[:n_seg*nperseg].reshape(n_seg, nperseg, 2)
        tgt_seg = tgt[:n_seg*nperseg].reshape(n_seg, nperseg, 2)
        nmse = NMSE(pred_seg, tgt_seg)
        evm = EVM(pred_seg, tgt_seg, sample_rate=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub, nperseg=nperseg)
        aclr_l, aclr_r = ACLR(pred_seg, fs=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub, nperseg=nperseg)
        aclr_avg = (aclr_l+aclr_r)/2
        n_params = sum(p.numel() for p in net_dpd.parameters())
        print(f"  k={k:3d} lr={lr:5s} | params={n_params} | NMSE={nmse:7.2f} EVM={evm:7.2f} ACLR_AVG={aclr_avg:7.2f}")
        rows.append({'k': k, 'lr': lr, 'params': n_params, 'NMSE': nmse, 'EVM': evm,
                     'ACLR_L': aclr_l, 'ACLR_R': aclr_r, 'ACLR_AVG': aclr_avg})

    # Merge in rev10d lr=1e-3 baselines
    for k in [21, 41]:
        sub = rev10d_csv[rev10d_csv['k'] == k]
        if len(sub):
            r = sub.iloc[0]
            rows.append({'k': k, 'lr': '1e-3', 'params': int(r['params']),
                         'NMSE': float(r['NMSE']), 'EVM': float(r['EVM']),
                         'ACLR_L': float(r['ACLR_L']), 'ACLR_R': float(r['ACLR_R']),
                         'ACLR_AVG': float(r['ACLR_AVG'])})
            print(f"  [from rev10d CSV] k={k:3d} lr=1e-3   | params={int(r['params'])} | NMSE={r['NMSE']:7.2f} EVM={r['EVM']:7.2f} ACLR_AVG={r['ACLR_AVG']:7.2f}")

    df = pd.DataFrame(rows)
    os.makedirs('results/rev10e_lr_collapse', exist_ok=True)
    df.to_csv('results/rev10e_lr_collapse/rev10e_results.csv', index=False, float_format='%.3f')
    print("\nSaved.")
    print(df.sort_values(['k', 'lr']).to_string(index=False))


if __name__ == '__main__':
    main()
