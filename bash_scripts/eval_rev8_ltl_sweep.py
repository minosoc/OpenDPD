"""rev8 sliding eval for DGRU + Transformer LTL FL sweep (and merges with rev6/rev6b FL=50)."""
import os, sys, json, glob
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import models as M
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR


def sliding_pred(net_cas, x, FL, device, batch=128):
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
        for s in range(0, M_out, batch):
            e = min(s+batch, M_out)
            idx = torch.arange(s, e).unsqueeze(1) + torch.arange(FL).unsqueeze(0)
            win = x_t[idx].to(device)
            out[s:e] = net_cas(win)[:, -1, :].cpu().numpy()
    return out


def build_dgru(ckpt):
    net = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
    net.load_state_dict(torch.load(ckpt, map_location='cpu')); net.eval()
    return net


def build_transformer(ckpt):
    net = M.CoreModel(input_size=2, hidden_size=6, num_layers=1, backbone_type='transformer',
                      n_heads=2, d_ff=18, use_pos_encoding=0)
    net.load_state_dict(torch.load(ckpt, map_location='cpu')); net.eval()
    return net


def main():
    device = torch.device('cuda:0')
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name='DPA_200MHz')
    G = float(np.max(np.abs(y_train[:,0]+1j*y_train[:,1])) / np.max(np.abs(X_train[:,0]+1j*X_train[:,1])))
    target_test = G * X_test
    with open('datasets/DPA_200MHz/spec.json') as f: spec = json.load(f)
    fs = float(spec['input_signal_fs']); bw_main = float(spec['bw_main_ch']); n_sub = int(spec['n_sub_ch'])
    nperseg = int(spec['nperseg'])
    pa_ckpt = 'save/DPA_200MHz/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt'

    configs = [
        # (label, build_fn, ckpt_glob_template)
        ('DGRU LTL', build_dgru,
         lambda FL: f'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_{FL}/DPD_S_0_M_DGRU_H_8_F_{FL}_P_*_LTL_1.pt'),
        ('Transformer V0 LTL', build_transformer,
         lambda FL: f'save/DPA_200MHz/train_dpd/PA_S_0_M_DGRU_H_8_F_{FL}/DPD_S_0_M_TRANSFORMER_H_6_F_{FL}_P_*_LTL_1_PE_0.pt'),
    ]

    rows = []
    for label, build_fn, ckpt_glob in configs:
        for FL in [50, 100, 200, 500, 1000]:
            cands = sorted(glob.glob(ckpt_glob(FL)))
            if not cands:
                print(f"SKIP {label} FL={FL}: no ckpt at {ckpt_glob(FL)}"); continue
            ckpt = cands[0]
            net_pa = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
            net_pa.load_state_dict(torch.load(pa_ckpt, map_location='cpu')); net_pa.eval()
            net_dpd = build_fn(ckpt)
            net_cas = M.CascadedModel(dpd_model=net_dpd, pa_model=net_pa).to(device).eval()
            pred = sliding_pred(net_cas, X_test, FL, device)
            tgt = target_test[FL-1:] if FL <= len(X_test) else target_test
            n_seg = pred.shape[0] // nperseg
            if n_seg < 1:
                print(f"SKIP {label} FL={FL}: pred too short"); continue
            pred_seg = pred[:n_seg*nperseg].reshape(n_seg, nperseg, 2)
            tgt_seg = tgt[:n_seg*nperseg].reshape(n_seg, nperseg, 2)
            nmse = NMSE(pred_seg, tgt_seg)
            evm = EVM(pred_seg, tgt_seg, sample_rate=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub, nperseg=nperseg)
            aclr_l, aclr_r = ACLR(pred_seg, fs=int(fs), bw_main_ch=bw_main, n_sub_ch=n_sub, nperseg=nperseg)
            aclr_avg = (aclr_l+aclr_r)/2
            n_params = sum(p.numel() for p in net_dpd.parameters())
            print(f"  {label:22s} FL={FL:5d} | params={n_params} | NMSE={nmse:7.2f} EVM={evm:7.2f} ACLR_AVG={aclr_avg:7.2f}")
            rows.append({'model': label, 'FL': FL, 'params': n_params, 'NMSE': nmse, 'EVM': evm,
                         'ACLR_L': aclr_l, 'ACLR_R': aclr_r, 'ACLR_AVG': aclr_avg})

    df = pd.DataFrame(rows)
    os.makedirs('results/rev8_all_ltl_sweep', exist_ok=True)
    df.to_csv('results/rev8_all_ltl_sweep/rev8_dgru_attn_ltl_results.csv', index=False, float_format='%.3f')
    print("\nSaved.")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
