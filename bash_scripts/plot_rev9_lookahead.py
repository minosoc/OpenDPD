"""rev9 plot: Lookahead M sweep at fixed FL_total=201 — DGRU/Transformer/Mamba.

X-axis = M (lookahead samples = future samples available at output time).
M=0 → pure causal. M=FL//2 → fully centered.
"""
import os, pandas as pd, matplotlib.pyplot as plt, matplotlib.ticker as mticker

OUTDIR = 'results/rev9_lookahead_sweep'
df = pd.read_csv(f'{OUTDIR}/rev9_lookahead_results.csv')

models = ['DGRU', 'TransformerV0', 'Mamba']
colors = {'DGRU': 'tab:blue', 'TransformerV0': 'tab:orange', 'Mamba': 'tab:green'}
markers = {'DGRU': 'd', 'TransformerV0': 's', 'Mamba': 'o'}
labels = {'DGRU': 'DGRU (486p)', 'TransformerV0': 'Transformer V0 no-PE (488p)', 'Mamba': 'Mamba (548p)'}

metrics = [('NMSE', 'NMSE (dB)'), ('EVM', 'EVM (dB)'), ('ACLR_AVG', 'ACLR_AVG (dB)')]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
for ax, (col, ylabel) in zip(axes, metrics):
    for m_name in models:
        sub = df[df['model'] == m_name].sort_values('M')
        if sub.empty: continue
        ax.plot(sub['M'], sub[col], f'{markers[m_name]}-', color=colors[m_name],
                lw=2.4, ms=10, label=labels[m_name])
        for _, r in sub.iterrows():
            ax.annotate(f"{r[col]:.1f}", (r['M'], r[col]), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=8.5,
                        color=colors[m_name], fontweight='bold')
    ax.set_xlabel('Lookahead M (future samples)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{col} vs Lookahead M (FL_total=201, LTL)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9.5)
    ax.axvline(0, color='gray', lw=0.8, ls=':', alpha=0.6)
fig.suptitle('Lookahead Sweep — GMP leading-term emulation in NN @ FL_total=201, DPA_200MHz',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev9_lookahead_sweep.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev9_lookahead_sweep.png")

# Pivot table
piv = df.pivot_table(index='M', columns='model', values=['NMSE', 'EVM', 'ACLR_AVG'])
piv.to_csv(f'{OUTDIR}/rev9_lookahead_summary.csv')
print("\nSummary (rows=M, columns=model):")
print(piv.round(2).to_string())
