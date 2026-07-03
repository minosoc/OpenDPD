"""Eval Mamba multi-seed variance on GaN_combined (overall NMSE).
seeds 1-4 from tagged ckpts; seed 0 from the original sweep CSV.
Reports per-FL mean/std/best/worst vs DGRU, and a scatter figure."""
import os, sys, glob
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import models as M
from modules.data_collector import load_dataset
from utils.metrics import NMSE

FLS = [50, 200, 1000]
SEEDS = [1, 2, 3, 4]
PA = 'save/GaN_combined/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt'
# seed-0 (original sweep) and DGRU baseline from combined_results.csv
csv = pd.read_csv('results/combined_ltl_sweep/combined_results.csv')
seed0 = {int(r.FL): r.NMSE_all for _, r in csv[csv.arch == 'Mamba'].iterrows()}
dgru = {int(r.FL): r.NMSE_all for _, r in csv[csv.arch == 'DGRU'].iterrows()}


def build_mamba(ck):
    n = M.CoreModel(input_size=2, hidden_size=6, num_layers=1, backbone_type='mamba',
                    mamba_d_state=4, mamba_d_conv=4, mamba_expand=2)
    n.load_state_dict(torch.load(ck, map_location='cpu')); n.eval(); return n


def slide(cas, x, FL, dev, batch=256):
    N = len(x); Mo = N - FL + 1; out = np.zeros((Mo, 2), np.float32); xt = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        for s in range(0, Mo, batch):
            e = min(s + batch, Mo); idx = torch.arange(s, e).unsqueeze(1) + torch.arange(FL).unsqueeze(0)
            out[s:e] = cas(xt[idx].to(dev))[:, -1, :].cpu().numpy()
    return out


def seg(a, n): m = len(a) // n; return a[:m * n].reshape(m, n, 2)


def main():
    dev = torch.device('cuda:0')
    Xtr, ytr, _, _, Xte, yte = load_dataset(dataset_name='GaN_combined')
    G = float(np.max(np.abs(ytr[:, 0] + 1j * ytr[:, 1])) / np.max(np.abs(Xtr[:, 0] + 1j * Xtr[:, 1])))
    tgt = G * Xte
    results = {FL: {0: seed0.get(FL, np.nan)} for FL in FLS}
    for FL in FLS:
        for sd in SEEDS:
            ck = f'save/GaN_combined/train_dpd/PA_S_0_M_DGRU_H_8_F_{FL}/DPD_S_0_M_MAMBA_H_6_F_{FL}_P_548_LTL_1_LR1e-3_CLIP200_SEED{sd}.pt'
            if not os.path.exists(ck):
                print(f"MISS FL={FL} seed={sd}"); results[FL][sd] = np.nan; continue
            pa = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
            pa.load_state_dict(torch.load(PA, map_location='cpu')); pa.eval()
            cas = M.CascadedModel(dpd_model=build_mamba(ck), pa_model=pa).to(dev).eval()
            pred = slide(cas, Xte, FL, dev); tg = tgt[FL - 1:FL - 1 + len(pred)]
            results[FL][sd] = NMSE(seg(pred, 2048), seg(tg, 2048))
            print(f"  FL={FL} seed={sd}: {results[FL][sd]:.2f}")

    print(f"\n{'FL':>6}{'mean':>8}{'std':>7}{'best':>8}{'worst':>8}{'DGRU':>8}  seeds")
    rows = []
    for FL in FLS:
        vals = np.array([results[FL][s] for s in [0] + SEEDS], float)
        v = vals[~np.isnan(vals)]
        print(f"{FL:>6}{v.mean():>8.1f}{v.std():>7.1f}{v.min():>8.1f}{v.max():>8.1f}{dgru[FL]:>8.1f}  "
              + " ".join(f"{x:.1f}" for x in vals))
        rows.append({'FL': FL, 'mean': v.mean(), 'std': v.std(), 'best': v.min(), 'worst': v.max(), 'DGRU': dgru[FL]})
    pd.DataFrame(rows).to_csv('results/combined_ltl_sweep/mamba_multiseed.csv', index=False, float_format='%.3f')

    # figure: scatter seeds + best/mean vs FL, DGRU line
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for FL in FLS:
        vals = [results[FL][s] for s in [0] + SEEDS if not np.isnan(results[FL][s])]
        ax.scatter([FL] * len(vals), vals, s=45, color='#27867a', alpha=0.6, zorder=3,
                   label='Mamba (seeds)' if FL == FLS[0] else None)
        ax.scatter([FL], [min(vals)], s=120, facecolors='none', edgecolors='#146356', lw=2, zorder=4,
                   label='Mamba best-of-5' if FL == FLS[0] else None)
    ax.plot(FLS, [dgru[FL] for FL in FLS], 'o-', color='#1f5fb4', lw=2.3, ms=9, label='DGRU', zorder=2)
    ax.set_xscale('log'); ax.set_xticks(FLS); ax.set_xticklabels(FLS)
    ax.set_xlabel('Frame length FL'); ax.set_ylabel('Overall NMSE (dB)')
    ax.set_title('Mamba seed variance vs DGRU', fontweight='bold'); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig('results/combined_ltl_sweep/mamba_multiseed.png', dpi=150, bbox_inches='tight')
    print("saved results/combined_ltl_sweep/mamba_multiseed.{csv,png}")


if __name__ == '__main__':
    main()
