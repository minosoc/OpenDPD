"""Zero-shot LOO eval: DPD trained on 3 conditions, tested on held-out condition.
Universal PA (all 4, frozen). Compare zero-shot vs oracle (model trained WITH the condition)."""
import os, sys, glob
import numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import models as M
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR

PA = 'save/GaN_combined/train_pa/PA_S_0_M_DGRU_H_8_F_50_P_486.pt'
# held -> (condition name in combined_results.csv, fs, bw)
HELD = {'c24': ('GaN_24dBm', 128e6, 20e6), 'c27': ('GaN_27dBm', 128e6, 20e6),
        'b20': ('GaN_bw100_20dBm', 640e6, 100e6), 'b24': ('GaN_bw100_24dBm', 640e6, 100e6)}
CONFIGS = [('dgru', 200), ('dgru', 1000), ('mamba', 1000)]
NPS = 2048
csv = pd.read_csv('results/combined_ltl_sweep/combined_results.csv')


def build(arch, ck):
    if arch == 'dgru':
        n = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
    else:
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


def seg(a): m = len(a) // NPS; return a[:m * NPS].reshape(m, NPS, 2)


def oracle(cond, arch, FL):
    a = 'DGRU' if arch == 'dgru' else 'Mamba'
    r = csv[(csv.arch == a) & (csv.FL == FL)]
    return float(r[f'NMSE_{cond}'].iloc[0]) if len(r) else np.nan


def main():
    dev = torch.device('cuda:0')
    rows = []
    for held, (cond, fs, bw) in HELD.items():
        Xtr, ytr, _, _, Xte, yte = load_dataset(dataset_name=f'GaN_loo_{held}')
        G = float(np.max(np.abs(ytr[:, 0] + 1j * ytr[:, 1])) / np.max(np.abs(Xtr[:, 0] + 1j * Xtr[:, 1])))
        tgt_full = G * Xte
        for arch, FL in CONFIGS:
            hs = 8 if arch == 'dgru' else 6
            pat = f'save/GaN_loo_{held}/train_dpd/PA_S_0_M_DGRU_H_8_F_{FL}/DPD_S_0_M_{arch.upper()}_H_{hs}_F_{FL}_P_*_LTL_1*.pt'
            cands = sorted(glob.glob(pat))
            if not cands:
                print(f"MISS {held} {arch} FL={FL}"); continue
            pa = M.CoreModel(input_size=2, hidden_size=8, num_layers=1, backbone_type='dgru')
            pa.load_state_dict(torch.load(PA, map_location='cpu')); pa.eval()
            cas = M.CascadedModel(dpd_model=build(arch, cands[0]), pa_model=pa).to(dev).eval()
            pred = slide(cas, Xte, FL, dev); tg = tgt_full[FL - 1:FL - 1 + len(pred)]
            ps, gs = seg(pred), seg(tg)
            nmse = NMSE(ps, gs)
            evm = EVM(ps, gs, sample_rate=int(fs), bw_main_ch=bw, n_sub_ch=10, nperseg=NPS)
            al, ar = ACLR(ps, fs=int(fs), bw_main_ch=bw, n_sub_ch=10, nperseg=NPS)
            orc = oracle(cond, arch, FL)
            rows.append(dict(held=cond, arch=arch, FL=FL, zs_NMSE=nmse, zs_EVM=evm, zs_ACLR=(al + ar) / 2,
                             oracle_NMSE=orc, gap=nmse - orc))
            print(f"  {cond:16s} {arch:5s} FL={FL:4d} | zero-shot NMSE={nmse:7.2f} EVM={evm:7.2f} ACLR={(al+ar)/2:7.2f} | oracle={orc:7.2f} gap={nmse-orc:+.1f}")

    df = pd.DataFrame(rows)
    df.to_csv('results/combined_ltl_sweep/loo_results.csv', index=False, float_format='%.3f')
    print("\n=== zero-shot NMSE by (arch,FL), mean over 4 held-out conditions ===")
    print(df.groupby(['arch', 'FL']).agg(zs_NMSE=('zs_NMSE', 'mean'), oracle=('oracle_NMSE', 'mean'),
                                         gap=('gap', 'mean')).round(2).to_string())
    print("\nSaved results/combined_ltl_sweep/loo_results.csv")


if __name__ == '__main__':
    main()
