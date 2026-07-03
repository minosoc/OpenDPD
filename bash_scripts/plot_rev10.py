"""rev10 plot: SwiGLU + GMP-stem Transformer vs V0 baseline + DGRU LTL @ FL=200."""
import os, pandas as pd, matplotlib.pyplot as plt
import numpy as np

OUTDIR = 'results/rev10_transformer_gmp'
df = pd.read_csv(f'{OUTDIR}/rev10_results.csv')

# Order: V0 → SwiGLU → GMP → DGRU
order = ['Transformer V0 LTL (rev8)',
         'rev10a SwiGLU d_ff=12',
         'rev10c GMP-stem k=5 d_ff=10',
         'DGRU LTL (rev8, ref)']
present = df['model'].tolist()
df = df.set_index('model').loc[[o for o in order if o in present]].reset_index()

colors = ['tab:orange', 'tab:purple', 'tab:red', 'tab:blue']
metrics = [('NMSE', 'NMSE (dB)'), ('EVM', 'EVM (dB)'), ('ACLR_AVG', 'ACLR_AVG (dB)')]

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
for ax, (col, ylabel) in zip(axes, metrics):
    bars = ax.bar(range(len(df)), df[col], color=colors[:len(df)], edgecolor='black')
    for i, (v, p) in enumerate(zip(df[col], df['params'])):
        ax.annotate(f"{v:.2f}", (i, v), textcoords='offset points',
                    xytext=(0, -16 if v < -5 else 6), ha='center', fontsize=10,
                    fontweight='bold', color='white' if v < -20 else 'black')
        ax.annotate(f"({p}p)", (i, 0.5), textcoords='offset points',
                    xytext=(0, 4), ha='center', fontsize=8, alpha=0.7)
    ax.set_xticks(range(len(df)))
    labels = ['V0\nbaseline', 'SwiGLU\nFFN', 'GMP-stem\nConv', 'DGRU\nref']
    ax.set_xticklabels(labels[:len(df)], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'{col} @ FL=200, LTL', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('rev10: Transformer FFN/Stem variants vs V0 baseline & DGRU @ DPA_200MHz, FL=200, LTL, sliding inference',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev10_transformer_gmp.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev10_transformer_gmp.png")
print(df.to_string(index=False))
