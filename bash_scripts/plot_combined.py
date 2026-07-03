"""Plot combined-data DPD sweep: DGRU/Transformer/Mamba vs FL.
Overall NMSE + mean per-condition EVM/ACLR, plus per-condition NMSE grid."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'combined_ltl_sweep')
df = pd.read_csv(f'{OUT}/combined_results.csv')

CONDS = ['GaN_24dBm', 'GaN_27dBm', 'GaN_bw100_20dBm', 'GaN_bw100_24dBm']
COL = {'DGRU': 'tab:blue', 'Transformer': 'tab:red', 'Mamba': 'tab:green'}
MK = {'DGRU': 'o', 'Transformer': 's', 'Mamba': '^'}
df['EVM_mean'] = df[[f'EVM_{c}' for c in CONDS]].mean(axis=1)
df['ACLR_mean'] = df[[f'ACLR_{c}' for c in CONDS]].mean(axis=1)

plt.rcParams.update({'font.size': 11, 'axes.spines.right': False, 'axes.spines.top': False})

# Figure 1: headline metrics vs FL
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for col, (key, ylab) in zip(ax, [('NMSE_all', 'Overall NMSE (dB)'),
                                  ('EVM_mean', 'Mean per-cond EVM (dB)'),
                                  ('ACLR_mean', 'Mean per-cond ACLR (dB)')]):
    for arch in COL:
        s = df[df['arch'] == arch].sort_values('FL')
        if len(s):
            col.plot(s['FL'], s[key], MK[arch] + '-', color=COL[arch], lw=2, ms=9, label=arch)
            for _, r in s.iterrows():
                col.annotate(f"{r[key]:.1f}", (r['FL'], r[key]), textcoords='offset points',
                             xytext=(0, 7), ha='center', fontsize=7.5, color=COL[arch])
    col.set_xscale('log'); col.set_xticks([50,100,200,500,1000]); col.set_xticklabels([50,100,200,500,1000])
    col.set_xlabel('Frame length FL (context)'); col.set_ylabel(ylab); col.grid(alpha=0.3)
    col.set_title(ylab.split(' (')[0], fontweight='bold'); col.legend()
fig.suptitle('Combined GaN (4 conditions pooled) — DPD LTL, architecture × FL sweep',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f'{OUT}/combined_headline.png', dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'saved {OUT}/combined_headline.png')

# Figure 2: per-condition NMSE grid
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for a, c in zip(ax.ravel(), CONDS):
    for arch in COL:
        s = df[df['arch'] == arch].sort_values('FL')
        if len(s):
            a.plot(s['FL'], s[f'NMSE_{c}'], MK[arch] + '-', color=COL[arch], lw=2, ms=8, label=arch)
    a.set_xscale('log'); a.set_xticks([50,100,200,500,1000]); a.set_xticklabels([50,100,200,500,1000])
    a.set_xlabel('FL'); a.set_ylabel('NMSE (dB)'); a.grid(alpha=0.3)
    a.set_title(c, fontweight='bold', loc='left'); a.legend(fontsize=9)
fig.suptitle('Combined-model per-condition NMSE (one model, evaluated on each condition)',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f'{OUT}/combined_percond_nmse.png', dpi=140, bbox_inches='tight'); plt.close(fig)
print(f'saved {OUT}/combined_percond_nmse.png')
