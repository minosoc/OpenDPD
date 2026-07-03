"""rev4 plot: Transformer variants vs DGRU at FL=50."""
import os, pandas as pd, matplotlib.pyplot as plt
import numpy as np

OUTDIR = 'results/rev4_attn_variants'
df = pd.read_csv(f'{OUTDIR}/rev4_attn_variants_results.csv')

# Order
order = ['DGRU H=8 (486p)', 'V0 baseline (488p)', 'V1 +residual concat (500p)', 'V2 +I/O MLP h=12 (704p)']
df = df.set_index('model').loc[order].reset_index()

colors = ['tab:blue', 'tab:gray', 'tab:orange', 'tab:red']
metrics = [('NMSE', 'NMSE (dB)'), ('EVM', 'EVM (dB)'), ('ACLR_AVG', 'ACLR_AVG (dB)')]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (col, ylabel) in zip(axes, metrics):
    bars = ax.bar(range(len(df)), df[col], color=colors, edgecolor='black')
    for i, (v, p) in enumerate(zip(df[col], df['dpd_params'])):
        ax.annotate(f"{v:.1f}", (i, v), textcoords='offset points', xytext=(0, -14 if v < -5 else 6),
                    ha='center', fontsize=10, fontweight='bold', color='white' if v < -20 else 'black')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([m.replace(' (', '\n(') for m in df['model']], rotation=0, ha='center', fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'{col} (FL=50)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Transformer Variants vs DGRU @ FL=50 — DPA_200MHz (paper config, sliding inference, nperseg=2560)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev4_attn_variants.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev4_attn_variants.png")
print(df.to_string(index=False))
