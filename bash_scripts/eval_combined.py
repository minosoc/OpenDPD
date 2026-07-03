"""Evaluate combined-data DPD models (DGRU/Transformer/Mamba x FL sweep, LTL)
trained on GaN_combined. Overall NMSE + per-condition NMSE/EVM/ACLR
(each condition uses its own fs/bw; boundary-crossing windows excluded)."""
import os, sys, json, glob
import numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import models as M
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR

FLS = [50, 100, 200, 500, 1000]
PA_CKPT = 'save/GaN_combined/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt'
DPD_DIR = lambda FL: f'save/GaN_combined/train_dpd/PA_S_0_M_DGRU_H_8_F_{FL}'
GLOBS = {
    'DGRU':        lambda FL: f'{DPD_DIR(FL)}/DPD_S_0_M_DGRU_H_8_F_{FL}_P_*_LTL_1.pt',
    'Transformer': lambda FL: f'{DPD_DIR(FL)}/DPD_S_0_M_TRANSFORMER_H_6_F_{FL}_P_*_LTL_1_PE_0.pt',
    'Mamba':       lambda FL: f'{DPD_DIR(FL)}/DPD_S_0_M_MAMBA_H_6_F_{FL}_P_*_LTL_1.pt',
}
# per-condition metric params (fs, bw, n_sub, nperseg)
COND = {
    'GaN_24dBm':       (128e6, 20e6, 10, 2048),
    'GaN_27dBm':       (128e6, 20e6, 10, 2048),
    'GaN_bw100_20dBm': (640e6, 100e6, 10, 2048),
    'GaN_bw100_24dBm': (640e6, 100e6, 10, 2048),
}


def build(arch, ckpt):
    if arch == 'DGRU':
        net = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
    elif arch == 'Transformer':
        net = M.CoreModel(input_size=2, hidden_size=6, num_layers=1, backbone_type='transformer',
                          n_heads=2, d_ff=18, use_pos_encoding=0)
    else:  # Mamba
        net = M.CoreModel(input_size=2, hidden_size=6, num_layers=1, backbone_type='mamba',
                          mamba_d_state=4, mamba_d_conv=4, mamba_expand=2)
    net.load_state_dict(torch.load(ckpt, map_location='cpu')); net.eval()
    return net


def sliding_pred(net_cas, x, FL, device, batch=256):
    N = x.shape[0]; M_out = N - FL + 1
    out = np.zeros((M_out, 2), dtype=np.float32)
    xt = torch.from_numpy(x.astype(np.float32)); net_cas.eval()
    with torch.no_grad():
        for s in range(0, M_out, batch):
            e = min(s + batch, M_out)
            idx = torch.arange(s, e).unsqueeze(1) + torch.arange(FL).unsqueeze(0)
            out[s:e] = net_cas(xt[idx].to(device))[:, -1, :].cpu().numpy()
    return out


def seg(a, nperseg):
    m = len(a) // nperseg
    return a[:m*nperseg].reshape(m, nperseg, 2) if m >= 1 else None


def main():
    device = torch.device('cuda:0')
    Xtr, ytr, _, _, Xte, yte = load_dataset(dataset_name='GaN_combined')
    G = float(np.max(np.abs(ytr[:,0]+1j*ytr[:,1])) / np.max(np.abs(Xtr[:,0]+1j*Xtr[:,1])))
    tgt_full = G * Xte
    src = pd.read_csv('datasets/GaN_combined/test_source.csv')['source'].values
    # condition blocks (contiguous) in test
    blocks = {}
    for c in COND:
        idx = np.where(src == c)[0]
        blocks[c] = (idx.min(), idx.max() + 1)  # [start, end)

    rows = []
    for arch, gf in GLOBS.items():
        for FL in FLS:
            cands = sorted(glob.glob(gf(FL)))
            if not cands:
                print(f"SKIP {arch} FL={FL}: no ckpt"); continue
            net_pa = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
            net_pa.load_state_dict(torch.load(PA_CKPT, map_location='cpu')); net_pa.eval()
            net_dpd = build(arch, cands[0])
            cas = M.CascadedModel(dpd_model=net_dpd, pa_model=net_pa).to(device).eval()
            pred = sliding_pred(cas, Xte, FL, device)            # pred[i] <-> sample i+FL-1
            tgt = tgt_full[FL-1:FL-1+len(pred)]
            nparam = sum(p.numel() for p in net_dpd.parameters())
            # overall NMSE
            ps, gs = seg(pred, 2048), seg(tgt, 2048)
            nmse_all = NMSE(ps, gs)
            row = {'arch': arch, 'FL': FL, 'params': nparam, 'NMSE_all': nmse_all}
            # per-condition (clean windows fully within block)
            for c, (s, e) in blocks.items():
                fs, bw, nsub, nps = COND[c]
                lo, hi = s, e - FL          # pred indices i in [lo, hi) keep window in block
                if hi - lo < nps:
                    row[f'NMSE_{c}'] = row[f'EVM_{c}'] = row[f'ACLR_{c}'] = np.nan; continue
                pc, gc = pred[lo:hi], tgt_full[lo+FL-1:hi+FL-1]
                pcs, gcs = seg(pc, nps), seg(gc, nps)
                row[f'NMSE_{c}'] = NMSE(pcs, gcs)
                row[f'EVM_{c}'] = EVM(pcs, gcs, sample_rate=int(fs), bw_main_ch=bw, n_sub_ch=nsub, nperseg=nps)
                al, ar = ACLR(pcs, fs=int(fs), bw_main_ch=bw, n_sub_ch=nsub, nperseg=nps)
                row[f'ACLR_{c}'] = (al + ar) / 2
            rows.append(row)
            print(f"  {arch:12s} FL={FL:5d} | params={nparam} | NMSE_all={nmse_all:7.2f} | "
                  + " ".join(f"{c.replace('GaN_','')}:{row[f'NMSE_{c}']:6.1f}" for c in COND))

    df = pd.DataFrame(rows)
    os.makedirs('results/combined_ltl_sweep', exist_ok=True)
    df.to_csv('results/combined_ltl_sweep/combined_results.csv', index=False, float_format='%.3f')
    print("\nSaved results/combined_ltl_sweep/combined_results.csv")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
