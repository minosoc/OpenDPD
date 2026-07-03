"""Plot frame-length sweep results — NMSE/EVM/ACLR vs FL for 3 backbones."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

df = pd.read_csv('bash_scripts/sliding_window_results.csv')
fls = sorted(df['frame_length'].unique())

backbones = [
    ('DGRU',        'tab:blue',   'o-'),
    ('ATTN(no PE)', 'tab:orange', 's-'),
    ('ATTN(PE)',    'tab:green',  '^-'),
]
metrics = [('NMSE', 'NMSE (dB)'), ('EVM', 'EVM (dB)'), ('ACLR_AVG', 'ACLR_AVG (dB)')]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (col, ylabel) in zip(axes, metrics):
    for bb, color, mk in backbones:
        sub = df[df['backbone'] == bb].sort_values('frame_length')
        ax.plot(sub['frame_length'], sub[col], mk, color=color, label=bb, lw=2, ms=8)
    ax.set_xlabel('Frame length (samples)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks(fls)
    ax.set_xticklabels([str(f) for f in fls], rotation=45)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title(f'{col} vs Frame Length', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)

fig.suptitle('DPA_200MHz Frame-Length Sweep — DGRU vs Transformer (no/with PE), ~486 params each, sliding-window eval',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('bash_scripts/framelen_sweep.png', dpi=130, bbox_inches='tight')
print("Saved: bash_scripts/framelen_sweep.png")

# Print summary table
print("\n=== Summary (TEST NMSE / EVM / ACLR_AVG, dB) ===")
print(f"{'FL':>6} | {'DGRU':>30} | {'ATTN(no PE)':>30} | {'ATTN(PE)':>30}")
print('-' * 110)
for fl in fls:
    row = ""
    for bb, _, _ in backbones:
        r = df[(df['backbone'] == bb) & (df['frame_length'] == fl)]
        if not r.empty:
            r = r.iloc[0]
            row += f" | NMSE={r['NMSE']:6.2f} EVM={r['EVM']:6.2f} ACLR={r['ACLR_AVG']:6.2f}"
        else:
            row += f" | {'-':>30}"
    print(f"{fl:>6}{row}")
