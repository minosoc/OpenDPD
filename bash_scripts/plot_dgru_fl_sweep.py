"""DGRU FL sweep visualization — NMSE/EVM/ACLR vs frame_length.

Paper config (FL=50,100,200,500,1000). Sliding inference + nperseg=2560 metric.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# DGRU H=8 results from sliding-window eval (paper config, nperseg=2560 metric)
data = pd.DataFrame([
    {'FL': 50,   'NMSE': -41.19, 'EVM': -43.68, 'ACLR_L': -48.75, 'ACLR_R': -45.37, 'ACLR_AVG': -47.06},
    {'FL': 100,  'NMSE': -42.59, 'EVM': -44.77, 'ACLR_L': -49.73, 'ACLR_R': -47.41, 'ACLR_AVG': -48.57},
    {'FL': 200,  'NMSE': -43.31, 'EVM': -45.29, 'ACLR_L': -48.62, 'ACLR_R': -47.30, 'ACLR_AVG': -47.96},
    {'FL': 500,  'NMSE': -43.38, 'EVM': -45.29, 'ACLR_L': -39.09, 'ACLR_R': -37.12, 'ACLR_AVG': -38.10},
    {'FL': 1000, 'NMSE': -44.31, 'EVM': -47.14, 'ACLR_L': -31.02, 'ACLR_R': -28.30, 'ACLR_AVG': -29.66},
])
data.to_csv('results/rev1_dgru_fl_sweep/rev1_dgru_fl_sweep_results.csv', index=False, float_format='%.3f')

fls = data['FL'].values

# === Figure 1: All 3 main metrics in one panel ===
fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
ax.plot(fls, data['NMSE'],     'o-', color='tab:blue',   lw=2.2, ms=10, label='NMSE')
ax.plot(fls, data['EVM'],      's-', color='tab:green',  lw=2.2, ms=9,  label='EVM')
ax.plot(fls, data['ACLR_AVG'], '^-', color='tab:red',    lw=2.2, ms=10, label='ACLR_AVG')
for i, fl in enumerate(fls):
    ax.annotate(f"{data['NMSE'].iloc[i]:.1f}",     (fl, data['NMSE'].iloc[i]),     textcoords='offset points', xytext=(0, 10),  ha='center', fontsize=9, color='tab:blue')
    ax.annotate(f"{data['EVM'].iloc[i]:.1f}",      (fl, data['EVM'].iloc[i]),      textcoords='offset points', xytext=(0, 10),  ha='center', fontsize=9, color='tab:green')
    ax.annotate(f"{data['ACLR_AVG'].iloc[i]:.1f}", (fl, data['ACLR_AVG'].iloc[i]), textcoords='offset points', xytext=(0, -16), ha='center', fontsize=9, color='tab:red')
ax.set_xscale('log')
ax.set_xticks(fls)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel('Frame Length (samples)', fontsize=12)
ax.set_ylabel('Value (dB) — lower is better', fontsize=12)
ax.set_title('DGRU H=8 (486 params) — Performance vs Frame Length\n(paper config, sliding inference, nperseg=2560 metric)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(-52, -25)
plt.tight_layout()
plt.savefig('results/rev1_dgru_fl_sweep/rev1_dgru_fl_sweep.png', dpi=140, bbox_inches='tight')
print("Saved results/rev1_dgru_fl_sweep/rev1_dgru_fl_sweep.png")

# === Figure 2: 2x2 panel — separate ACLR_L/R ===
fig2, axes = plt.subplots(2, 2, figsize=(13, 8))
metrics_panel = [
    ('NMSE',     'tab:blue',  axes[0, 0], 'NMSE (dB)'),
    ('EVM',      'tab:green', axes[0, 1], 'EVM (dB)'),
    ('ACLR_AVG', 'tab:red',   axes[1, 0], 'ACLR_AVG (dB)'),
    (None,       None,        axes[1, 1], 'ACLR_L vs ACLR_R'),
]
for m, color, ax, title in metrics_panel:
    if m is None:
        ax.plot(fls, data['ACLR_L'], 'o-', color='tab:purple', lw=2.2, ms=9, label='ACLR_L')
        ax.plot(fls, data['ACLR_R'], 's-', color='tab:orange', lw=2.2, ms=9, label='ACLR_R')
        ax.legend(loc='best', fontsize=11)
    else:
        ax.plot(fls, data[m], 'o-', color=color, lw=2.2, ms=10)
        for i, fl in enumerate(fls):
            ax.annotate(f"{data[m].iloc[i]:.2f}", (fl, data[m].iloc[i]), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9)
    ax.set_xscale('log')
    ax.set_xticks(fls)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Frame Length (samples)')
    ax.set_ylabel(title)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

fig2.suptitle('DGRU H=8 (486 params) FL Sweep — DPA_200MHz (sliding inference, nperseg=2560)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/rev1_dgru_fl_sweep/rev1_dgru_fl_sweep_4panel.png', dpi=140, bbox_inches='tight')
print("Saved results/rev1_dgru_fl_sweep/rev1_dgru_fl_sweep_4panel.png")

# === Print summary ===
print("\nSummary (TEST, dB):")
print(data.to_string(index=False))
