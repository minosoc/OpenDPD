"""rev3 (b): Superplot — DGRU vs Transformer no-PE vs Transformer w/PE FL sweep.

All three at ~486-488 params, paper config, sliding inference, nperseg=2560 metric.
"""
import os, pandas as pd, matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTDIR = 'results/rev3b_all_backbones_fl_sweep'
os.makedirs(OUTDIR, exist_ok=True)

# Combine rev1 (DGRU), rev3 (ATTN no-PE), rev2 (ATTN w/PE) results
data_all = {
    'DGRU H=8 (486 params)':                pd.DataFrame([
        {'FL': 50,   'NMSE': -41.19, 'EVM': -43.68, 'ACLR_L': -48.75, 'ACLR_R': -45.37, 'ACLR_AVG': -47.06},
        {'FL': 100,  'NMSE': -42.59, 'EVM': -44.77, 'ACLR_L': -49.73, 'ACLR_R': -47.41, 'ACLR_AVG': -48.57},
        {'FL': 200,  'NMSE': -43.31, 'EVM': -45.29, 'ACLR_L': -48.62, 'ACLR_R': -47.30, 'ACLR_AVG': -47.96},
        {'FL': 500,  'NMSE': -43.38, 'EVM': -45.29, 'ACLR_L': -39.09, 'ACLR_R': -37.12, 'ACLR_AVG': -38.10},
        {'FL': 1000, 'NMSE': -44.31, 'EVM': -47.14, 'ACLR_L': -31.02, 'ACLR_R': -28.30, 'ACLR_AVG': -29.66},
    ]),
    'Transformer no-PE (488 params)':       pd.DataFrame([
        {'FL': 50,   'NMSE': -21.23, 'EVM': -21.80, 'ACLR_L': -47.60, 'ACLR_R': -45.33, 'ACLR_AVG': -46.47},
        {'FL': 100,  'NMSE': -21.10, 'EVM': -21.60, 'ACLR_L': -47.25, 'ACLR_R': -45.34, 'ACLR_AVG': -46.29},
        {'FL': 200,  'NMSE': -21.05, 'EVM': -21.59, 'ACLR_L': -47.48, 'ACLR_R': -45.82, 'ACLR_AVG': -46.65},
        {'FL': 500,  'NMSE': -21.04, 'EVM': -21.55, 'ACLR_L': -40.16, 'ACLR_R': -36.63, 'ACLR_AVG': -38.39},
        {'FL': 1000, 'NMSE': -21.06, 'EVM': -21.53, 'ACLR_L': -32.48, 'ACLR_R': -28.00, 'ACLR_AVG': -30.24},
    ]),
    'Transformer w/PE (488 params)':        pd.DataFrame([
        {'FL': 50,   'NMSE': -21.01, 'EVM': -21.85, 'ACLR_L': -37.10, 'ACLR_R': -36.21, 'ACLR_AVG': -36.66},
        {'FL': 100,  'NMSE': -21.38, 'EVM': -22.11, 'ACLR_L': -38.64, 'ACLR_R': -38.02, 'ACLR_AVG': -38.33},
        {'FL': 200,  'NMSE':  -6.28, 'EVM':  -8.53, 'ACLR_L': -16.08, 'ACLR_R': -14.65, 'ACLR_AVG': -15.36},
        {'FL': 500,  'NMSE': -19.56, 'EVM': -21.19, 'ACLR_L': -38.48, 'ACLR_R': -35.27, 'ACLR_AVG': -36.87},
        {'FL': 1000, 'NMSE': -20.25, 'EVM': -21.17, 'ACLR_L': -32.78, 'ACLR_R': -27.45, 'ACLR_AVG': -30.12},
    ]),
}

# Save combined CSV
rows = []
for tag, df in data_all.items():
    for _, r in df.iterrows():
        rows.append({'backbone': tag, **r.to_dict()})
combined = pd.DataFrame(rows)
combined.to_csv(f'{OUTDIR}/rev3b_all_backbones_results.csv', index=False, float_format='%.3f')

# Color/marker scheme
styles = {
    'DGRU H=8 (486 params)':           ('tab:blue',   'o', '-'),
    'Transformer no-PE (488 params)':  ('tab:orange', 's', '-'),
    'Transformer w/PE (488 params)':   ('tab:green',  '^', '-'),
}

# === Figure 1: 3-panel comparison ===
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
metrics = [('NMSE', 'NMSE (dB)'), ('EVM', 'EVM (dB)'), ('ACLR_AVG', 'ACLR_AVG (dB)')]
for ax, (col, ylabel) in zip(axes, metrics):
    for tag, df in data_all.items():
        c, m, ls = styles[tag]
        ax.plot(df['FL'], df[col], marker=m, color=c, linestyle=ls, lw=2.2, ms=9, label=tag)
    ax.set_xscale('log')
    fls = data_all['DGRU H=8 (486 params)']['FL'].values
    ax.set_xticks(fls); ax.xaxis.set_major_formatter(mticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Frame Length (samples)'); ax.set_ylabel(ylabel)
    ax.set_title(f'{col} vs Frame Length', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=9)
fig.suptitle('All Backbones FL Sweep — DPA_200MHz (paper config, ~486-488 params, sliding inference, nperseg=2560)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev3b_all_backbones_fl_sweep.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev3b_all_backbones_fl_sweep.png")

# === Figure 2: 4-panel detailed ===
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
panels = [
    ('NMSE', axes2[0, 0]),
    ('EVM', axes2[0, 1]),
    ('ACLR_AVG', axes2[1, 0]),
    (None, axes2[1, 1]),  # ACLR_L vs ACLR_R per backbone
]
for col, ax in panels:
    if col is None:
        for tag, df in data_all.items():
            c, m, ls = styles[tag]
            ax.plot(df['FL'], df['ACLR_L'], marker=m, color=c, linestyle='--', lw=1.6, ms=7, label=f'{tag} — ACLR_L')
            ax.plot(df['FL'], df['ACLR_R'], marker=m, color=c, linestyle=':',  lw=1.6, ms=7, label=f'{tag} — ACLR_R')
        ax.set_title('ACLR_L (dashed) vs ACLR_R (dotted)', fontweight='bold')
        ax.set_ylabel('ACLR (dB)')
        ax.legend(loc='best', fontsize=7, ncol=2)
    else:
        for tag, df in data_all.items():
            c, m, ls = styles[tag]
            ax.plot(df['FL'], df[col], marker=m, color=c, linestyle=ls, lw=2.2, ms=9, label=tag)
        ax.set_ylabel(f'{col} (dB)'); ax.set_title(f'{col} (dB)', fontweight='bold')
        ax.legend(loc='best', fontsize=9)
    ax.set_xscale('log')
    fls = data_all['DGRU H=8 (486 params)']['FL'].values
    ax.set_xticks(fls); ax.xaxis.set_major_formatter(mticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Frame Length (samples)')
    ax.grid(True, alpha=0.3, which='both')
fig2.suptitle('All Backbones FL Sweep (detailed) — DPA_200MHz', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev3b_all_backbones_fl_sweep_4panel.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev3b_all_backbones_fl_sweep_4panel.png")

# Print summary table
print("\n=== Combined Summary ===")
print(combined.to_string(index=False))
