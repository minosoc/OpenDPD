"""rev7 plot: Mamba FL sweep with DGRU + Transformer overlay."""
import os, pandas as pd, matplotlib.pyplot as plt, matplotlib.ticker as mticker

OUTDIR = 'results/rev7_mamba_fl_sweep'
mamba = pd.read_csv(f'{OUTDIR}/rev7_mamba_results.csv')

# rev1 DGRU + rev3 Transformer no-PE baselines
dgru = pd.DataFrame([
    {'FL': 50,   'NMSE': -41.19, 'EVM': -43.68, 'ACLR_AVG': -47.06},
    {'FL': 100,  'NMSE': -42.59, 'EVM': -44.77, 'ACLR_AVG': -48.57},
    {'FL': 200,  'NMSE': -43.31, 'EVM': -45.29, 'ACLR_AVG': -47.96},
    {'FL': 500,  'NMSE': -43.38, 'EVM': -45.29, 'ACLR_AVG': -38.10},
    {'FL': 1000, 'NMSE': -44.31, 'EVM': -47.14, 'ACLR_AVG': -29.66},
])
attn = pd.DataFrame([
    {'FL': 50,   'NMSE': -21.23, 'EVM': -21.80, 'ACLR_AVG': -46.47},
    {'FL': 100,  'NMSE': -21.10, 'EVM': -21.60, 'ACLR_AVG': -46.29},
    {'FL': 200,  'NMSE': -21.05, 'EVM': -21.59, 'ACLR_AVG': -46.65},
    {'FL': 500,  'NMSE': -21.04, 'EVM': -21.55, 'ACLR_AVG': -38.39},
    {'FL': 1000, 'NMSE': -21.06, 'EVM': -21.53, 'ACLR_AVG': -30.24},
])

fls = sorted(mamba['FL'].unique())
metrics = [('NMSE', 'NMSE (dB)'), ('EVM', 'EVM (dB)'), ('ACLR_AVG', 'ACLR_AVG (dB)')]

# === Figure 1: 3-panel with overlay ===
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
for ax, (col, ylabel) in zip(axes, metrics):
    ax.plot(dgru['FL'], dgru[col],   'd-', color='tab:blue',   lw=2.2, ms=10, label='DGRU H=8 (486p)')
    ax.plot(attn['FL'], attn[col],   's-', color='tab:orange', lw=2.2, ms=9,  label='Transformer no-PE (488p)')
    ax.plot(mamba['FL'], mamba[col], 'o-', color='tab:green',  lw=2.5, ms=11, label='Mamba (548p)')
    for _, r in mamba.iterrows():
        ax.annotate(f"{r[col]:.1f}", (r['FL'], r[col]), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9, color='tab:green', fontweight='bold')
    ax.set_xscale('log'); ax.set_xticks(fls)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Frame Length (samples)'); ax.set_ylabel(ylabel)
    ax.set_title(f'{col} vs Frame Length', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10)
fig.suptitle('Mamba vs DGRU vs Transformer FL Sweep — DPA_200MHz (paper config, sliding inference, nperseg=2560)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev7_mamba_fl_sweep.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev7_mamba_fl_sweep.png")

# === Figure 2: 4-panel with ACLR_L/R ===
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
for col, ax in [('NMSE', axes2[0,0]), ('EVM', axes2[0,1]), ('ACLR_AVG', axes2[1,0])]:
    ax.plot(dgru['FL'], dgru[col],   'd-', color='tab:blue',   lw=2.2, ms=10, label='DGRU (486p)')
    ax.plot(attn['FL'], attn[col],   's-', color='tab:orange', lw=2.2, ms=9,  label='Transformer (488p)')
    ax.plot(mamba['FL'], mamba[col], 'o-', color='tab:green',  lw=2.5, ms=11, label='Mamba (548p)')
    ax.set_xscale('log'); ax.set_xticks(fls)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Frame Length'); ax.set_ylabel(f'{col} (dB)')
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both'); ax.legend(loc='best', fontsize=9)

# ACLR_L vs ACLR_R (Mamba only)
ax = axes2[1,1]
ax.plot(mamba['FL'], mamba['ACLR_L'], 'o-', color='tab:purple', lw=2, ms=9, label='Mamba ACLR_L')
ax.plot(mamba['FL'], mamba['ACLR_R'], 's-', color='tab:red',    lw=2, ms=9, label='Mamba ACLR_R')
ax.set_xscale('log'); ax.set_xticks(fls)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel('Frame Length'); ax.set_ylabel('ACLR (dB)')
ax.set_title('Mamba ACLR_L vs ACLR_R', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both'); ax.legend(loc='best', fontsize=10)

fig2.suptitle('Mamba FL Sweep — detailed comparison vs DGRU/Transformer', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev7_mamba_fl_sweep_4panel.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev7_mamba_fl_sweep_4panel.png")
print(mamba.to_string(index=False))
