"""rev8 plot: All architectures (DGRU/Transformer/Mamba) under LTL — FL sweep."""
import os, pandas as pd, matplotlib.pyplot as plt, matplotlib.ticker as mticker

OUTDIR = 'results/rev8_all_ltl_sweep'

# Load rev8 DGRU LTL + Transformer LTL FL sweep
df_rev8 = pd.read_csv(f'{OUTDIR}/rev8_dgru_attn_ltl_results.csv')
dgru = df_rev8[df_rev8['model'] == 'DGRU LTL'].sort_values('FL').reset_index(drop=True)
attn = df_rev8[df_rev8['model'] == 'Transformer V0 LTL'].sort_values('FL').reset_index(drop=True)

# Load rev7b Mamba LTL FL sweep
mamba = pd.read_csv('results/rev7b_mamba_ltl_sweep/rev7b_mamba_ltl_results.csv').sort_values('FL').reset_index(drop=True)

fls = sorted(set(dgru['FL']).union(attn['FL']).union(mamba['FL']))
metrics = [('NMSE', 'NMSE (dB)'), ('EVM', 'EVM (dB)'), ('ACLR_AVG', 'ACLR_AVG (dB)')]

# === Figure: 3-panel ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
for ax, (col, ylabel) in zip(axes, metrics):
    ax.plot(dgru['FL'],  dgru[col],  'd-', color='tab:blue',   lw=2.4, ms=11, label=f'DGRU LTL ({int(dgru["params"].iloc[0])}p)')
    ax.plot(attn['FL'],  attn[col],  's-', color='tab:orange', lw=2.4, ms=10, label=f'Transformer V0 LTL ({int(attn["params"].iloc[0])}p)')
    ax.plot(mamba['FL'], mamba[col], 'o-', color='tab:green',  lw=2.6, ms=11, label=f'Mamba LTL ({int(mamba["params"].iloc[0])}p)')
    for df_, color, dy in [(dgru, 'tab:blue', 10), (attn, 'tab:orange', -14), (mamba, 'tab:green', 10)]:
        for _, r in df_.iterrows():
            ax.annotate(f"{r[col]:.1f}", (r['FL'], r[col]), textcoords='offset points',
                        xytext=(0, dy), ha='center', fontsize=8.5, color=color, fontweight='bold')
    ax.set_xscale('log'); ax.set_xticks(fls)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Frame Length (samples)'); ax.set_ylabel(ylabel)
    ax.set_title(f'{col} vs Frame Length (LTL)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=9.5)
fig.suptitle('All architectures @ LTL — DPA_200MHz (paper config, sliding inference, nperseg=2560)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev8_all_ltl_sweep.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev8_all_ltl_sweep.png")

# === Combined wide table ===
wide = pd.DataFrame({'FL': fls})
for name, d in [('DGRU_LTL', dgru), ('TransformerV0_LTL', attn), ('Mamba_LTL', mamba)]:
    for col in ['NMSE', 'EVM', 'ACLR_AVG']:
        wide[f'{name}_{col}'] = wide['FL'].map(dict(zip(d['FL'], d[col]))).round(2)
wide.to_csv(f'{OUTDIR}/rev8_all_ltl_summary.csv', index=False)
print("\nSummary (all LTL):")
print(wide.to_string(index=False))
