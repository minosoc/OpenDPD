"""rev10d plot: GMP-stem kernel sweep + V0 + DGRU references."""
import os, pandas as pd, matplotlib.pyplot as plt, matplotlib.ticker as mticker

OUTDIR = 'results/rev10d_gmp_kernel_sweep'
df = pd.read_csv(f'{OUTDIR}/rev10d_results.csv').sort_values('k').reset_index(drop=True)

# Reference baselines @ FL=200 LTL
V0 = {'NMSE': -21.04, 'EVM': -21.58, 'ACLR_AVG': -43.06}
DGRU = {'NMSE': -32.00, 'EVM': -34.59, 'ACLR_AVG': -44.56}

metrics = [('NMSE', 'NMSE (dB)'), ('EVM', 'EVM (dB)'), ('ACLR_AVG', 'ACLR_AVG (dB)')]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
for ax, (col, ylabel) in zip(axes, metrics):
    ax.plot(df['k'], df[col], 'o-', color='tab:red', lw=2.5, ms=11, label='GMP-stem k sweep (d_ff=10)')
    for _, r in df.iterrows():
        ax.annotate(f"{r[col]:.1f}", (r['k'], r[col]), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9, color='tab:red', fontweight='bold')
    ax.axhline(V0[col], ls='--', color='tab:orange', lw=2, label=f'Transformer V0 (488p): {V0[col]:.1f}')
    ax.axhline(DGRU[col], ls='--', color='tab:blue', lw=2, label=f'DGRU LTL (486p): {DGRU[col]:.1f}')
    ax.set_xscale('log'); ax.set_xticks(df['k'])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('GMP-stem kernel size k')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{col} vs kernel size', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=9)
fig.suptitle('rev10d GMP-stem kernel sweep — ACLR vs k @ FL=200, LTL, DPA_200MHz',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev10d_kernel_sweep.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev10d_kernel_sweep.png")
print(df.to_string(index=False))
