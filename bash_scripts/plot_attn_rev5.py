"""rev5 plot: All Transformer variants vs DGRU at FL=50."""
import os, pandas as pd, matplotlib.pyplot as plt
import numpy as np

OUTDIR = 'results/rev5_attn_advanced'
df = pd.read_csv(f'{OUTDIR}/rev5_attn_advanced_results.csv')

# Order — DGRU first, then V0 baseline, then rev4 variants, then rev5 variants
order = ['DGRU H=8', 'V0 baseline',
         'rev4 V1 +residual concat', 'rev4 V2 +I/O MLP h=12',
         'rev5_d Conv stem k=3', 'rev5_e Local attn w=7',
         'rev5_f L=2', 'rev5_g n_heads=1']
df = df.set_index('model').loc[order].reset_index()

colors = ['tab:blue', 'tab:gray',
          'tab:cyan', 'tab:cyan',
          'tab:red', 'tab:red',
          'tab:purple', 'tab:purple']

metrics = [('NMSE', 'NMSE (dB)'), ('EVM', 'EVM (dB)'), ('ACLR_AVG', 'ACLR_AVG (dB)')]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (col, ylabel) in zip(axes, metrics):
    bars = ax.bar(range(len(df)), df[col], color=colors, edgecolor='black')
    for i, (v, p) in enumerate(zip(df[col], df['params'])):
        ax.annotate(f"{v:.1f}", (i, v), textcoords='offset points',
                    xytext=(0, -14 if v < -5 else 6), ha='center', fontsize=10,
                    fontweight='bold', color='white' if v < -25 else 'black')
        ax.annotate(f"({p}p)", (i, 0.5), textcoords='offset points',
                    xytext=(0, 4), ha='center', fontsize=8, alpha=0.7)
    ax.set_xticks(range(len(df)))
    labels = [m.replace(' +', '\n+').replace('rev5_', 'rev5_').replace('rev4 ', 'rev4 ') for m in df['model']]
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'{col} (FL=50)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(df.loc[0, col], linestyle='--', color='tab:blue', alpha=0.4, lw=1)

fig.suptitle('All Transformer Variants vs DGRU @ FL=50 — DPA_200MHz (paper config, sliding inference, nperseg=2560)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev5_attn_advanced.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev5_attn_advanced.png")
print(df.to_string(index=False))
