"""rev7b plot: Mamba LTL FL sweep vs rev6 LTL baselines (DGRU LTL, Transformer V0 LTL).

LTL baselines only have FL=50. Mamba LTL covers FL=50/100/200/500/1000.
Faint dashed lines show full-loss reference (rev1 DGRU, rev3 Transformer no-PE, rev7 Mamba full).
"""
import os, pandas as pd, matplotlib.pyplot as plt, matplotlib.ticker as mticker

OUTDIR = 'results/rev7b_mamba_ltl_sweep'
mamba_ltl = pd.read_csv(f'{OUTDIR}/rev7b_mamba_ltl_results.csv')

# rev6 LTL baselines (FL=50 only)
dgru_ltl_csv = pd.read_csv('results/rev6_dgru_last_token_loss/rev6_dgru_ltl_results.csv')
attn_ltl_csv = pd.read_csv('results/rev6b_attn_last_token_loss/rev6b_attn_ltl_results.csv')
dgru_ltl = dgru_ltl_csv[dgru_ltl_csv['model'].str.contains('LTL', na=False)].iloc[0]
attn_ltl = attn_ltl_csv[attn_ltl_csv['model'] == 'Transformer V0 LTL'].iloc[0]

# Full-loss FL sweep references (for faint dashed context)
dgru_full = pd.DataFrame([
    {'FL': 50,   'NMSE': -41.19, 'EVM': -43.68, 'ACLR_AVG': -47.06},
    {'FL': 100,  'NMSE': -42.59, 'EVM': -44.77, 'ACLR_AVG': -48.57},
    {'FL': 200,  'NMSE': -43.31, 'EVM': -45.29, 'ACLR_AVG': -47.96},
    {'FL': 500,  'NMSE': -43.38, 'EVM': -45.29, 'ACLR_AVG': -38.10},
    {'FL': 1000, 'NMSE': -44.31, 'EVM': -47.14, 'ACLR_AVG': -29.66},
])
attn_full = pd.DataFrame([
    {'FL': 50,   'NMSE': -21.23, 'EVM': -21.80, 'ACLR_AVG': -46.47},
    {'FL': 100,  'NMSE': -21.10, 'EVM': -21.60, 'ACLR_AVG': -46.29},
    {'FL': 200,  'NMSE': -21.05, 'EVM': -21.59, 'ACLR_AVG': -46.65},
    {'FL': 500,  'NMSE': -21.04, 'EVM': -21.55, 'ACLR_AVG': -38.39},
    {'FL': 1000, 'NMSE': -21.06, 'EVM': -21.53, 'ACLR_AVG': -30.24},
])
mamba_full = pd.read_csv('results/rev7_mamba_fl_sweep/rev7_mamba_results.csv')

fls = sorted(mamba_ltl['FL'].unique())
metrics = [('NMSE', 'NMSE (dB)'), ('EVM', 'EVM (dB)'), ('ACLR_AVG', 'ACLR_AVG (dB)')]

# === Figure: 3-panel LTL comparison + full-loss context ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
for ax, (col, ylabel) in zip(axes, metrics):
    # Faint full-loss references
    ax.plot(dgru_full['FL'], dgru_full[col],  '--', color='tab:blue',   lw=1.4, alpha=0.35, label='DGRU full-loss (rev1, ref)')
    ax.plot(attn_full['FL'], attn_full[col],  '--', color='tab:orange', lw=1.4, alpha=0.35, label='Transformer full-loss (rev3, ref)')
    ax.plot(mamba_full['FL'], mamba_full[col],'--', color='tab:green',  lw=1.4, alpha=0.35, label='Mamba full-loss (rev7, ref)')
    # LTL: solid bold
    ax.plot(mamba_ltl['FL'], mamba_ltl[col], 'o-', color='tab:green',  lw=2.6, ms=11, label='Mamba LTL (rev7b, 548p)')
    ax.scatter([50], [dgru_ltl[col]], marker='*', s=320, color='tab:blue',   edgecolor='black', lw=1.2, zorder=5, label='DGRU LTL (rev6, 486p)')
    ax.scatter([50], [attn_ltl[col]], marker='*', s=320, color='tab:orange', edgecolor='black', lw=1.2, zorder=5, label='Transformer V0 LTL (rev6b, 488p)')
    for _, r in mamba_ltl.iterrows():
        ax.annotate(f"{r[col]:.1f}", (r['FL'], r[col]), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9, color='tab:green', fontweight='bold')
    ax.annotate(f"{dgru_ltl[col]:.1f}", (50, dgru_ltl[col]), textcoords='offset points',
                xytext=(-22, -8), ha='right', fontsize=9, color='tab:blue', fontweight='bold')
    ax.annotate(f"{attn_ltl[col]:.1f}", (50, attn_ltl[col]), textcoords='offset points',
                xytext=(-22, -8), ha='right', fontsize=9, color='tab:orange', fontweight='bold')
    ax.set_xscale('log'); ax.set_xticks(fls)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Frame Length (samples)'); ax.set_ylabel(ylabel)
    ax.set_title(f'{col} vs Frame Length (LTL)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=8)
fig.suptitle('Mamba LTL FL Sweep vs LTL Baselines (rev6/rev6b) — DPA_200MHz (paper config, sliding inference, nperseg=2560)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev7b_mamba_ltl_sweep.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev7b_mamba_ltl_sweep.png")
print("\nMamba LTL results:")
print(mamba_ltl.to_string(index=False))
print(f"\nDGRU LTL @ FL=50:   NMSE={dgru_ltl['NMSE']:.2f}  EVM={dgru_ltl['EVM']:.2f}  ACLR_AVG={dgru_ltl['ACLR_AVG']:.2f}")
print(f"Trans LTL @ FL=50:  NMSE={attn_ltl['NMSE']:.2f}  EVM={attn_ltl['EVM']:.2f}  ACLR_AVG={attn_ltl['ACLR_AVG']:.2f}")
