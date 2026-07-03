"""rev6b plot: Transformer full vs LTL bar chart."""
import os, pandas as pd, matplotlib.pyplot as plt

OUTDIR = 'results/rev6b_attn_last_token_loss'
df = pd.read_csv(f'{OUTDIR}/rev6b_attn_ltl_results.csv')

colors = ['tab:blue', 'tab:gray', 'tab:orange']
metrics = [('NMSE', 'NMSE (dB)'), ('EVM', 'EVM (dB)'), ('ACLR_AVG', 'ACLR_AVG (dB)')]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
for ax, (col, ylabel) in zip(axes, metrics):
    bars = ax.bar(range(len(df)), df[col], color=colors, edgecolor='black')
    for i, v in enumerate(df[col]):
        ax.annotate(f"{v:.2f}", (i, v), textcoords='offset points',
                    xytext=(0, -14 if v < -5 else 6), ha='center', fontsize=11,
                    fontweight='bold', color='white' if v < -25 else 'black')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['model'], rotation=12, ha='right', fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'{col} (FL=50)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Transformer V0 — Full-position loss vs Last-token loss @ FL=50',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/rev6b_attn_ltl.png', dpi=140, bbox_inches='tight')
print(f"Saved {OUTDIR}/rev6b_attn_ltl.png")
print(df.to_string(index=False))
