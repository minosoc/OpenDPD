from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
# Non-interactive 백엔드 사용 (파일로 저장)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the file
# csv_path = "./pa_out/rev7_APA_200MHz/PA_S_0_M_DGRU_H_23_L_1_P_2751.csv"                             # rev7
# csv_path = "./pa_out/rev21_APA_200MHz/PA_S_0_M_TRANSFORMER_ENCODER_D_128_H_4_L_1_P_207170.csv"    # rev21
csv_path = "./pa_out/rev24_APA_200MHz/PA_S_0_M_TRANSFORMER_ENCODER_D_128_H_4_L_1_P_207170.csv"    # rev24
# csv_path = "./pa_out/rev26_APA_200MHz/PA_S_0_M_TRANSFORMER_ENCODER_D_128_H_4_L_1_P_207170.csv"    # rev26
df = pd.read_csv(csv_path)

# Helper to compute AM-AM and AM-PM
def compute_am_am_pm(Iin, Qin, Iout, Qout):
    in_mag = np.sqrt(Iin**2 + Qin**2)
    out_mag = np.sqrt(Iout**2 + Qout**2)
    in_phase = np.arctan2(Qin, Iin)
    out_phase = np.arctan2(Qout, Iout)
    am_pm = out_phase - in_phase
    am_pm = (am_pm + np.pi) % (2*np.pi) - np.pi
    return in_mag, out_mag, am_pm

# (i) I_out, Q_out
in_mag1, out_mag1, am_pm1 = compute_am_am_pm(
    df["I_in"], df["Q_in"], df["I_out"], df["Q_out"]
)

# (ii) I_out_actual, Q_out_actual
in_mag2, out_mag2, am_pm2 = compute_am_am_pm(
    df["I_in"], df["Q_in"], df["I_out_actual"], df["Q_out_actual"]
)

# 모든 플롯을 한 번에 표시 (2x2 subplot)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot (i) AM-AM
axes[0, 0].scatter(in_mag1, out_mag1, s=1)
axes[0, 0].set_xlabel("Input Amplitude")
axes[0, 0].set_ylabel("Output Amplitude")
axes[0, 0].set_title("AM-AM (I_out, Q_out)")
axes[0, 0].grid(True, alpha=0.3)

# Plot (i) AM-PM
axes[0, 1].scatter(in_mag1, am_pm1, s=1)
axes[0, 1].set_xlabel("Input Amplitude")
axes[0, 1].set_ylabel("Phase Difference (rad)")
axes[0, 1].set_title("AM-PM (I_out, Q_out)")
axes[0, 1].grid(True, alpha=0.3)

# Plot (ii) AM-AM
axes[1, 0].scatter(in_mag2, out_mag2, s=1)
axes[1, 0].set_xlabel("Input Amplitude")
axes[1, 0].set_ylabel("Output Amplitude")
axes[1, 0].set_title("AM-AM (I_out_actual, Q_out_actual)")
axes[1, 0].grid(True, alpha=0.3)

# Plot (ii) AM-PM
axes[1, 1].scatter(in_mag2, am_pm2, s=1)
axes[1, 1].set_xlabel("Input Amplitude")
axes[1, 1].set_ylabel("Phase Difference (rad)")
axes[1, 1].set_title("AM-PM (I_out_actual, Q_out_actual)")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# CSV 파일과 같은 디렉토리에 PNG 파일로 저장
csv_path_obj = Path(csv_path)
output_path = csv_path_obj.parent / "am_am_pm_plots.png"
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Plots saved to: {output_path}")
plt.close()
