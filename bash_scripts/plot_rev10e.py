"""rev10e plot: k=21 vs k=41 across lr ∈ {1e-3, 3e-4, 1e-4}."""
import os, pandas as pd, matplotlib.pyplot as plt
import numpy as np

OUTDIR = 'results/rev10e_lr_collapse'
df = pd.read_csv(f'{OUTDIR}/rev10e_results.csv', dtype={'lr': str})

# x-axis: lr (log scale), one line per k
df['lr_num'] = df['lr'].astype(float)
df = df.sort_values('lr_num', ascending=False)

ks = sorted(df['k'].unique())
colors = {21: 'tab:red', 41: 'tab:purple'}
markers = {21: 'o', 41: 's'}

metrics = [('NMSE', 'NMSE (dB)'), ('EVM', 'EVM (dB)'), ('ACLR_AVG', 'ACLR_AVG (dB)')]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
for ax, (col, ylabel) in zip(axes, metrics):
    for k in ks:
        sub = df[df['k'] == k].sort_values('lr_num', ascending=False)
        ax.plot(sub['lr_num'], sub[col], f'{markers[k]}-', color=colors[k],
                lw=2.4, ms=11, label=f'GMP-stem k={k}')
        for _, r in sub.iterrows():
            ax.annotate(f"{r[col]:.1f}", (r['lr_num'], r[col]), textcoords='offset points',
                        xytext=(0, 10), ha='center', fontsize=9, color=colors[k], fontweight='bold')
    ax.set_xscale('log'); ax.invert_xaxis()
    ax.set_xticks([1e-3, 5e-4, 3e-4, 1e-4])
    ax.set_xticklabels(['1e-3', '5e-4', '3e-4', '1e-4'])
    ax.set_xlabel('Learning rate (log scale, ↓)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{col} vs lr', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
fig.suptitle('rev10e — Did smaller lr rescue k=41 collapse? @ FL=200, LTL, DPA_200MHz',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev10e_lr_collapse.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev10e_lr_collapse.png")
print(df.to_string(index=False))
