"""rev3 (c) sliding-window eval for Transformer no-PE at d_model=6, 16, 32.

For each (d_model, FL), uses FL-window sliding stride=1 inference, then chunks
to nperseg=2560 for metric computation.
"""
import os, sys, glob, json, time
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import models as M
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR


def get_target_gain(x, y):
    return float(np.max(np.abs(x[:,0]+1j*x[:,1])) / np.max(np.abs(x[:,0]+1j*x[:,1])) * np.max(np.abs(y[:,0]+1j*y[:,1])) / np.max(np.abs(x[:,0]+1j*x[:,1])))


def sliding_cascade(net_cas, x, FL, device, batch=256):
    N = x.shape[0]
    if FL > N:
        with torch.no_grad():
            xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
            return net_cas(xt)[0].cpu().numpy()
    M_out = N - FL + 1
    out = np.zeros((M_out, 2), dtype=np.float32)
    x_t = torch.from_numpy(x.astype(np.float32))
    net_cas.eval()
    with torch.no_grad():
        for start in range(0, M_out, batch):
            end = min(start+batch, M_out)
            idx = torch.arange(start, end).unsqueeze(1) + torch.arange(FL).unsqueeze(0)
            win = x_t[idx].to(device)
            cas = net_cas(win)
            out[start:end] = cas[:, -1, :].cpu().numpy()
    return out


def main():
    device = torch.device('cuda:0')
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name='DPA_200MHz')
    in_amp = np.abs(X_train[:,0]+1j*X_train[:,1])
    out_amp = np.abs(y_train[:,0]+1j*y_train[:,1])
    G = float(np.max(out_amp) / np.max(in_amp))
    print(f"target_gain = {G:.4f}")
    with open('datasets/DPA_200MHz/spec.json') as f: spec = json.load(f)
    fs = float(spec['input_signal_fs']); bw_main = float(spec['bw_main_ch']); n_sub = int(spec['n_sub_ch'])
    nperseg_metric = int(spec['nperseg'])

    target_test = G * X_test
    pa_ckpt = 'save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt'

    configs = [
        (6,  18,  488),
        (16, 48, 2898),
        (32, 96, 10914),
    ]
    fls = [50, 100, 200, 500, 1000]
    rows = []
    for DM, DFF, P in configs:
        for FL in fls:
            dpd_ckpt = f'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_{FL}/DPD_S_0_M_TRANSFORMER_H_{DM}_F_{FL}_P_{P}_PE_0.pt'
            if not os.path.exists(dpd_ckpt):
                print(f"SKIP d_model={DM} FL={FL}: ckpt missing")
                continue
            net_pa = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
            net_pa.load_state_dict(torch.load(pa_ckpt, map_location='cpu')); net_pa.eval()
            net_dpd = M.CoreModel(input_size=2, hidden_size=DM, num_layers=1, backbone_type='transformer',
                                  n_heads=2, d_ff=DFF, use_pos_encoding=False)
            sd = torch.load(dpd_ckpt, map_location='cpu')
            sd = {k: v for k, v in sd.items() if not k.endswith('pos_encoding.pe')}
            net_dpd.load_state_dict(sd, strict=False); net_dpd.eval()
            net_cas = M.CascadedModel(dpd_model=net_dpd, pa_model=net_pa).to(device).eval()
            pred = sliding_cascade(net_cas, X_test, FL, device, batch=128)
            target_aligned = target_test[FL-1:] if FL <= len(X_test) else target_test
            n_seg = pred.shape[0] // nperseg_metric
            if n_seg < 1:
                print(f"SKIP d_model={DM} FL={FL}: pred too short")
                continue
            pred_seg = pred[:n_seg*nperseg_metric].reshape(n_seg, nperseg_metric, 2)
            tgt_seg = target_aligned[:n_seg*nperseg_metric].reshape(n_seg, nperseg_metric, 2)
            nmse = NMSE(pred_seg, tgt_seg)
            evm = EVM(pred_seg, tgt_seg, sample_rate=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub, nperseg=nperseg_metric)
            aclr_l, aclr_r = ACLR(pred_seg, fs=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub, nperseg=nperseg_metric)
            aclr_avg = (aclr_l+aclr_r)/2
            print(f"  d_model={DM:2d} ({P:5d}p) FL={FL:5d}: NMSE={nmse:7.2f}  EVM={evm:7.2f}  ACLR_L={aclr_l:7.2f}  ACLR_R={aclr_r:7.2f}  ACLR_AVG={aclr_avg:7.2f}")
            rows.append({'d_model': DM, 'params': P, 'FL': FL, 'NMSE': nmse, 'EVM': evm,
                         'ACLR_L': aclr_l, 'ACLR_R': aclr_r, 'ACLR_AVG': aclr_avg, 'n_seg': n_seg})
    df = pd.DataFrame(rows)
    os.makedirs('results/rev3c_attn_capacity_fl_sweep', exist_ok=True)
    df.to_csv('results/rev3c_attn_capacity_fl_sweep/rev3c_attn_capacity_results.csv', index=False, float_format='%.3f')
    print("\nSaved results/rev3c_attn_capacity_fl_sweep/rev3c_attn_capacity_results.csv")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
