"""rev6c: LTL-vs-LTL comparison — DGRU LTL vs Transformer V0 LTL.

Also adds full-loss results as faint background reference.
"""
import os, pandas as pd, matplotlib.pyplot as plt
import numpy as np

OUTDIR = 'results/rev6c_ltl_compare'
os.makedirs(OUTDIR, exist_ok=True)

# Compiled from rev1 (DGRU full), rev6 (DGRU LTL), rev4 V0 (ATTN full), rev6b (ATTN LTL)
data = pd.DataFrame([
    {'model': 'DGRU H=8',          'loss': 'full-position', 'NMSE': -41.19, 'EVM': -43.68, 'ACLR_AVG': -47.06},
    {'model': 'DGRU H=8',          'loss': 'last-token',    'NMSE': -31.36, 'EVM': -35.00, 'ACLR_AVG': -45.03},
    {'model': 'Transformer V0',    'loss': 'full-position', 'NMSE': -21.23, 'EVM': -21.80, 'ACLR_AVG': -46.47},
    {'model': 'Transformer V0',    'loss': 'last-token',    'NMSE': -21.31, 'EVM': -21.77, 'ACLR_AVG': -41.65},
])
data.to_csv(f'{OUTDIR}/rev6c_ltl_compare_results.csv', index=False, float_format='%.3f')

# === Figure 1: LTL only — head-to-head ===
ltl = data[data['loss'] == 'last-token']
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
colors = ['tab:blue', 'tab:orange']
for ax, col in zip(axes, ['NMSE', 'EVM', 'ACLR_AVG']):
    bars = ax.bar(range(len(ltl)), ltl[col], color=colors, edgecolor='black')
    for i, v in enumerate(ltl[col].values):
        ax.annotate(f"{v:.2f}", (i, v), textcoords='offset points',
                    xytext=(0, -16 if v < -5 else 6), ha='center',
                    fontsize=12, fontweight='bold',
                    color='white' if v < -25 else 'black')
    ax.set_xticks(range(len(ltl)))
    ax.set_xticklabels(ltl['model'].values, fontsize=11)
    ax.set_ylabel(f'{col} (dB)', fontsize=11)
    ax.set_title(f'{col} (LTL training)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
fig.suptitle('Last-Token Loss: DGRU H=8 vs Transformer V0 (FL=50, paper config except LTL)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev6c_ltl_only.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev6c_ltl_only.png")

# === Figure 2: full vs LTL side-by-side per model ===
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5.5))
models = ['DGRU H=8', 'Transformer V0']
loss_types = ['full-position', 'last-token']
width = 0.35
x = np.arange(len(models))
for ax, col in zip(axes2, ['NMSE', 'EVM', 'ACLR_AVG']):
    for j, loss in enumerate(loss_types):
        vals = [data[(data['model']==m) & (data['loss']==loss)][col].iloc[0] for m in models]
        c = 'tab:gray' if loss=='full-position' else 'tab:red'
        bars = ax.bar(x + (j - 0.5)*width, vals, width, color=c, edgecolor='black', label=loss)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.1f}", (x[i]+(j-0.5)*width, v), textcoords='offset points',
                        xytext=(0, -14 if v < -5 else 6), ha='center',
                        fontsize=9, fontweight='bold',
                        color='white' if v < -25 else 'black')
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel(f'{col} (dB)', fontsize=11)
    ax.set_title(f'{col} — full vs LTL', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=10)
fig2.suptitle('Full-position vs Last-token Loss for DGRU and Transformer V0 (FL=50)',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev6c_full_vs_ltl.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev6c_full_vs_ltl.png")
print(data.to_string(index=False))
