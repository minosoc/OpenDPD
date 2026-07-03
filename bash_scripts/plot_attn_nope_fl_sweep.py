"""rev3 (a): Transformer no-PE FL sweep visualization.

Paper config (FL=50,100,200,500,1000). Sliding inference + nperseg=2560 metric.
"""
import os, pandas as pd, matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTDIR = 'results/rev3_attn_nope_fl_sweep'
os.makedirs(OUTDIR, exist_ok=True)

data = pd.DataFrame([
    {'FL': 50,   'NMSE': -21.23, 'EVM': -21.80, 'ACLR_L': -47.60, 'ACLR_R': -45.33, 'ACLR_AVG': -46.47},
    {'FL': 100,  'NMSE': -21.10, 'EVM': -21.60, 'ACLR_L': -47.25, 'ACLR_R': -45.34, 'ACLR_AVG': -46.29},
    {'FL': 200,  'NMSE': -21.05, 'EVM': -21.59, 'ACLR_L': -47.48, 'ACLR_R': -45.82, 'ACLR_AVG': -46.65},
    {'FL': 500,  'NMSE': -21.04, 'EVM': -21.55, 'ACLR_L': -40.16, 'ACLR_R': -36.63, 'ACLR_AVG': -38.39},
    {'FL': 1000, 'NMSE': -21.06, 'EVM': -21.53, 'ACLR_L': -32.48, 'ACLR_R': -28.00, 'ACLR_AVG': -30.24},
])
data.to_csv(f'{OUTDIR}/rev3_attn_nope_fl_sweep_results.csv', index=False, float_format='%.3f')

fls = data['FL'].values

# 1-panel
fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
ax.plot(fls, data['NMSE'],     'o-', color='tab:blue',   lw=2.2, ms=10, label='NMSE')
ax.plot(fls, data['EVM'],      's-', color='tab:green',  lw=2.2, ms=9,  label='EVM')
ax.plot(fls, data['ACLR_AVG'], '^-', color='tab:red',    lw=2.2, ms=10, label='ACLR_AVG')
for i, fl in enumerate(fls):
    ax.annotate(f"{data['NMSE'].iloc[i]:.1f}",     (fl, data['NMSE'].iloc[i]),     textcoords='offset points', xytext=(0, 10),  ha='center', fontsize=9, color='tab:blue')
    ax.annotate(f"{data['EVM'].iloc[i]:.1f}",      (fl, data['EVM'].iloc[i]),      textcoords='offset points', xytext=(0, 10),  ha='center', fontsize=9, color='tab:green')
    ax.annotate(f"{data['ACLR_AVG'].iloc[i]:.1f}", (fl, data['ACLR_AVG'].iloc[i]), textcoords='offset points', xytext=(0, -16), ha='center', fontsize=9, color='tab:red')
ax.set_xscale('log'); ax.set_xticks(fls)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel('Frame Length (samples)', fontsize=12)
ax.set_ylabel('Value (dB) — lower is better', fontsize=12)
ax.set_title('Transformer no-PE (d=6, h=2, d_ff=18, L=1, 488 params) — Performance vs FL\n(paper config, sliding inference, nperseg=2560)',
             fontsize=11.5, fontweight='bold')
ax.grid(True, alpha=0.3, which='both'); ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(-52, -15)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev3_attn_nope_fl_sweep.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev3_attn_nope_fl_sweep.png")

# 4-panel
fig2, axes = plt.subplots(2, 2, figsize=(13, 8))
for m, color, ax, title in [
    ('NMSE', 'tab:blue', axes[0,0], 'NMSE (dB)'),
    ('EVM', 'tab:green', axes[0,1], 'EVM (dB)'),
    ('ACLR_AVG', 'tab:red', axes[1,0], 'ACLR_AVG (dB)'),
    (None, None, axes[1,1], 'ACLR_L vs ACLR_R'),
]:
    if m is None:
        ax.plot(fls, data['ACLR_L'], 'o-', color='tab:purple', lw=2.2, ms=9, label='ACLR_L')
        ax.plot(fls, data['ACLR_R'], 's-', color='tab:orange', lw=2.2, ms=9, label='ACLR_R')
        ax.legend(loc='best', fontsize=11)
    else:
        ax.plot(fls, data[m], 'o-', color=color, lw=2.2, ms=10)
        for i, fl in enumerate(fls):
            ax.annotate(f"{data[m].iloc[i]:.2f}", (fl, data[m].iloc[i]), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9)
    ax.set_xscale('log'); ax.set_xticks(fls)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Frame Length (samples)'); ax.set_ylabel(title)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
fig2.suptitle('Transformer no-PE (488 params) FL Sweep — DPA_200MHz', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev3_attn_nope_fl_sweep_4panel.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev3_attn_nope_fl_sweep_4panel.png")
print(data.to_string(index=False))
