"""rev3 (c) plot: Transformer no-PE capacity scaling FL sweep.

Compares d_model = 6 (488p) / 16 (2898p) / 32 (10914p) across FL=50,100,200,500,1000.
Also overlays DGRU H=8 (486p) as reference baseline.
"""
import os, pandas as pd, matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTDIR = 'results/rev3c_attn_capacity_fl_sweep'
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(f'{OUTDIR}/rev3c_attn_capacity_results.csv')

# DGRU baseline (from rev1) for comparison
dgru = pd.DataFrame([
    {'FL': 50,   'NMSE': -41.19, 'EVM': -43.68, 'ACLR_AVG': -47.06},
    {'FL': 100,  'NMSE': -42.59, 'EVM': -44.77, 'ACLR_AVG': -48.57},
    {'FL': 200,  'NMSE': -43.31, 'EVM': -45.29, 'ACLR_AVG': -47.96},
    {'FL': 500,  'NMSE': -43.38, 'EVM': -45.29, 'ACLR_AVG': -38.10},
    {'FL': 1000, 'NMSE': -44.31, 'EVM': -47.14, 'ACLR_AVG': -29.66},
])

styles = {
    6:  ('tab:gray',    'o', '-', 'Transformer d=6 (488p)'),
    16: ('tab:orange',  's', '-', 'Transformer d=16 (2898p)'),
    32: ('tab:red',     '^', '-', 'Transformer d=32 (10914p)'),
}

fls = sorted(df['FL'].unique())

# === Figure 1: 3-panel — NMSE / EVM / ACLR ===
fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
for ax, (col, ylabel) in zip(axes, [('NMSE', 'NMSE (dB)'), ('EVM', 'EVM (dB)'), ('ACLR_AVG', 'ACLR_AVG (dB)')]):
    # DGRU baseline (dashed gray as reference)
    ax.plot(dgru['FL'], dgru[col], 'd--', color='tab:blue', lw=2, ms=8, alpha=0.7, label='DGRU H=8 (486p) [ref]')
    # Transformer at varying d_model
    for dm in [6, 16, 32]:
        sub = df[df['d_model'] == dm].sort_values('FL')
        c, m, ls, label = styles[dm]
        ax.plot(sub['FL'], sub[col], marker=m, color=c, linestyle=ls, lw=2.2, ms=9, label=label)
    ax.set_xscale('log'); ax.set_xticks(fls)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Frame Length (samples)'); ax.set_ylabel(ylabel)
    ax.set_title(f'{col} vs Frame Length', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=9)
fig.suptitle('Transformer (no PE) Capacity Scaling FL Sweep — DPA_200MHz (paper config, sliding inference, nperseg=2560)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev3c_attn_capacity_fl_sweep.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev3c_attn_capacity_fl_sweep.png")

# === Figure 2: 4-panel detailed ===
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
panels = [('NMSE', axes2[0,0]), ('EVM', axes2[0,1]), ('ACLR_AVG', axes2[1,0]), (None, axes2[1,1])]
for col, ax in panels:
    if col is None:
        for dm in [6, 16, 32]:
            sub = df[df['d_model'] == dm].sort_values('FL')
            c, m, ls, label = styles[dm]
            ax.plot(sub['FL'], sub['ACLR_L'], marker=m, color=c, linestyle='--', lw=1.6, ms=7, label=f'd={dm} ACLR_L')
            ax.plot(sub['FL'], sub['ACLR_R'], marker=m, color=c, linestyle=':',  lw=1.6, ms=7, label=f'd={dm} ACLR_R')
        ax.set_title('ACLR_L (dashed) vs ACLR_R (dotted)', fontweight='bold')
        ax.set_ylabel('ACLR (dB)')
        ax.legend(loc='best', fontsize=7, ncol=2)
    else:
        ax.plot(dgru['FL'], dgru[col], 'd--', color='tab:blue', lw=2, ms=8, alpha=0.7, label='DGRU H=8 (486p)')
        for dm in [6, 16, 32]:
            sub = df[df['d_model'] == dm].sort_values('FL')
            c, m, ls, label = styles[dm]
            ax.plot(sub['FL'], sub[col], marker=m, color=c, linestyle=ls, lw=2.2, ms=9, label=label)
        ax.set_ylabel(f'{col} (dB)'); ax.set_title(f'{col} (dB)', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
    ax.set_xscale('log'); ax.set_xticks(fls)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Frame Length (samples)')
    ax.grid(True, alpha=0.3, which='both')
fig2.suptitle('Transformer Capacity Scaling FL Sweep (detailed) — DPA_200MHz', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev3c_attn_capacity_fl_sweep_4panel.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev3c_attn_capacity_fl_sweep_4panel.png")

print(df.to_string(index=False))
