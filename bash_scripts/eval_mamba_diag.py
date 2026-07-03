"""Eval Mamba instability-diagnostic ckpts (FL 50,200 x variants) on GaN_combined.
Overall NMSE + per-condition NMSE; compare vs ORIG baseline and DGRU."""
import os, sys, glob
import numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import models as M
from modules.data_collector import load_dataset
from utils.metrics import NMSE

COND = {'GaN_24dBm': 2048, 'GaN_27dBm': 2048, 'GaN_bw100_20dBm': 2048, 'GaN_bw100_24dBm': 2048}
PA = 'save/GaN_combined/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt'
VARIANTS = [('baseline lr1e-3', '_ORIG'),
            ('lr3e-4',          '_LR3e-4_CLIP200_SEED0'),
            ('clip1.0',         '_LR1e-3_CLIP1.0_SEED0'),
            ('seed1',           '_LR1e-3_CLIP200_SEED1')]


def build_mamba(ck):
    net = M.CoreModel(input_size=2, hidden_size=6, num_layers=1, backbone_type='mamba',
                      mamba_d_state=4, mamba_d_conv=4, mamba_expand=2)
    net.load_state_dict(torch.load(ck, map_location='cpu')); net.eval(); return net


def slide(cas, x, FL, device, batch=256):
    N = x.shape[0]; Mo = N - FL + 1; out = np.zeros((Mo, 2), np.float32)
    xt = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        for s in range(0, Mo, batch):
            e = min(s+batch, Mo); idx = torch.arange(s, e).unsqueeze(1) + torch.arange(FL).unsqueeze(0)
            out[s:e] = cas(xt[idx].to(device))[:, -1, :].cpu().numpy()
    return out


def segn(a, n): m = len(a)//n; return a[:m*n].reshape(m, n, 2)


def main():
    device = torch.device('cuda:0')
    Xtr, ytr, _, _, Xte, yte = load_dataset(dataset_name='GaN_combined')
    G = float(np.max(np.abs(ytr[:,0]+1j*ytr[:,1]))/np.max(np.abs(Xtr[:,0]+1j*Xtr[:,1])))
    tgt = G*Xte
    src = pd.read_csv('datasets/GaN_combined/test_source.csv')['source'].values
    blk = {c: (np.where(src==c)[0].min(), np.where(src==c)[0].max()+1) for c in COND}
    dgru_ref = {50: -44.28, 200: -44.90}  # from combined_results.csv

    rows = []
    for FL in [50, 200]:
        base = f'save/GaN_combined/train_dpd/PA_S_0_M_DGRU_H_8_F_{FL}/DPD_S_0_M_MAMBA_H_6_F_{FL}_P_548_LTL_1'
        for name, suf in VARIANTS:
            ck = base + suf + '.pt'
            if not os.path.exists(ck): print(f"SKIP FL={FL} {name}: {ck} missing"); continue
            pa = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
            pa.load_state_dict(torch.load(PA, map_location='cpu')); pa.eval()
            cas = M.CascadedModel(dpd_model=build_mamba(ck), pa_model=pa).to(device).eval()
            pred = slide(cas, Xte, FL, device); tg = tgt[FL-1:FL-1+len(pred)]
            nmse_all = NMSE(segn(pred, 2048), segn(tg, 2048))
            pc = {}
            for c, (s, e) in blk.items():
                lo, hi = s, e-FL
                pc[c] = NMSE(segn(pred[lo:hi], 2048), segn(tgt[lo+FL-1:hi+FL-1], 2048)) if hi-lo >= 2048 else np.nan
            rows.append({'FL': FL, 'variant': name, 'NMSE_all': nmse_all, **{f'NMSE_{c}': pc[c] for c in COND}})
            print(f"  FL={FL} {name:16s} NMSE_all={nmse_all:7.2f}  (DGRU ref {dgru_ref[FL]})")

    df = pd.DataFrame(rows)
    os.makedirs('results/combined_ltl_sweep', exist_ok=True)
    df.to_csv('results/combined_ltl_sweep/mamba_diag_results.csv', index=False, float_format='%.3f')
    print("\n" + df.to_string(index=False))
    print(f"\nDGRU baseline: FL=50 {dgru_ref[50]} | FL=200 {dgru_ref[200]}")


if __name__ == '__main__':
    main()
