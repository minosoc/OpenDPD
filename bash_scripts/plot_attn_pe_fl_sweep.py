"""rev2: Transformer w/PE FL sweep visualization — NMSE/EVM/ACLR vs frame_length.

Paper config (FL=50,100,200,500,1000). Sliding inference + nperseg=2560 metric.
Same setup as rev1 (DGRU) but with Transformer (d_model=6, n_heads=2, d_ff=18, L=1, w/ sinusoidal PE).
"""
import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTDIR = 'results/rev2_attn_pe_fl_sweep'
os.makedirs(OUTDIR, exist_ok=True)

# Transformer w/PE (d_model=6, n_heads=2, d_ff=18, L=1, ~488 params) results
# from sliding-window eval (paper config, nperseg=2560 metric)
data = pd.DataFrame([
    {'FL': 50,   'NMSE': -21.01, 'EVM': -21.85, 'ACLR_L': -37.10, 'ACLR_R': -36.21, 'ACLR_AVG': -36.66},
    {'FL': 100,  'NMSE': -21.38, 'EVM': -22.11, 'ACLR_L': -38.64, 'ACLR_R': -38.02, 'ACLR_AVG': -38.33},
    {'FL': 200,  'NMSE':  -6.28, 'EVM':  -8.53, 'ACLR_L': -16.08, 'ACLR_R': -14.65, 'ACLR_AVG': -15.36},
    {'FL': 500,  'NMSE': -19.56, 'EVM': -21.19, 'ACLR_L': -38.48, 'ACLR_R': -35.27, 'ACLR_AVG': -36.87},
    {'FL': 1000, 'NMSE': -20.25, 'EVM': -21.17, 'ACLR_L': -32.78, 'ACLR_R': -27.45, 'ACLR_AVG': -30.12},
])
data.to_csv(f'{OUTDIR}/rev2_attn_pe_fl_sweep_results.csv', index=False, float_format='%.3f')

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
ax.set_title('Transformer w/PE (d=6, h=2, d_ff=18, L=1, 488 params) — Performance vs Frame Length\n(paper config, sliding inference, nperseg=2560 metric)',
             fontsize=11.5, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(-45, 5)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev2_attn_pe_fl_sweep.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev2_attn_pe_fl_sweep.png")

# === Figure 2: 2x2 panel ===
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

fig2.suptitle('Transformer w/PE (488 params) FL Sweep — DPA_200MHz (sliding inference, nperseg=2560)',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev2_attn_pe_fl_sweep_4panel.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev2_attn_pe_fl_sweep_4panel.png")

print("\nSummary (TEST, dB):")
print(data.to_string(index=False))
