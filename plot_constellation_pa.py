#!/usr/bin/env python3
"""
Script to plot constellation diagram from CSV file containing I/Q data.

This script reads a PA output CSV file and plots a constellation diagram showing:
- PA model output (I_out/Q_out)
- Actual PA output (I_out_actual/Q_out_actual)

For randomly selected rows, showing the comparison between model and actual PA outputs.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

# CSV 파일 경로 지정
# csv_path = "./pa_out/rev7_APA_200MHz/PA_S_0_M_DGRU_H_23_L_1_P_2751.csv"                             # rev7
# csv_path = "./pa_out/rev21_APA_200MHz/PA_S_0_M_TRANSFORMER_ENCODER_D_128_H_4_L_1_P_207170.csv"    # rev21
csv_path = "./pa_out/rev24_APA_200MHz/PA_S_0_M_TRANSFORMER_ENCODER_D_128_H_4_L_1_P_207170.csv"    # rev24
# csv_path = "./pa_out/rev26_APA_200MHz/PA_S_0_M_TRANSFORMER_ENCODER_D_128_H_4_L_1_P_207170.csv"    # rev26


def normalize_constellation(I, Q, normalization='rms'):
    """
    Normalize I/Q data for constellation diagram display.
    
    Parameters:
    - I, Q: I and Q component arrays
    - normalization: 'rms' for RMS normalization, 'max' for max normalization, 'none' for no normalization
    
    Returns:
    - Normalized I and Q arrays
    """
    if normalization == 'rms':
        # RMS normalization
        rms = np.sqrt(np.mean(I**2 + Q**2))
        if rms > 0:
            I_norm = I / rms
            Q_norm = Q / rms
        else:
            I_norm = I
            Q_norm = Q
    elif normalization == 'max':
        # Max magnitude normalization
        max_mag = np.max(np.sqrt(I**2 + Q**2))
        if max_mag > 0:
            I_norm = I / max_mag
            Q_norm = Q / max_mag
        else:
            I_norm = I
            Q_norm = Q
    else:
        I_norm = I
        Q_norm = Q
    
    return I_norm, Q_norm


def plot_constellation_from_csv(csv_path, n_samples=64, seed=None, alpha=0.7, 
                                normalization='rms', point_size=50, line_alpha=0.3):
    """
    Read PA output CSV file and plot constellation diagram showing model vs actual comparison.
    
    Selects n_samples random rows and plots two points for each:
    - PA model output (I_out/Q_out) - green
    - Actual PA output (I_out_actual/Q_out_actual) - blue
    
    Parameters:
    - csv_path: Path to CSV file
    - n_samples: Number of rows to randomly select (default: 64)
    - seed: Random seed for reproducibility
    - alpha: Transparency for scatter points
    - normalization: 'rms', 'max', or 'none' for normalization method
    - point_size: Size of scatter points
    - line_alpha: Transparency for connecting lines
    
    Returns:
    - fig: matplotlib figure object
    - output_path: Path to saved PNG file
    """
    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ['I_out', 'Q_out', 'I_out_actual', 'Q_out_actual']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. "
                        f"Found columns: {df.columns.tolist()}")
    
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Select random rows using numpy random choice
    total_rows = len(df)
    n_samples = min(n_samples, total_rows)
    
    # Random sampling without replacement
    selected_indices = np.random.choice(total_rows, size=n_samples, replace=False)
    selected_indices = np.sort(selected_indices)  # Sort for better visualization
    
    print(f"Selected {n_samples} random rows out of {total_rows} total rows")
    
    # Extract I/Q data for selected rows
    I_out = df.loc[selected_indices, 'I_out'].values
    Q_out = df.loc[selected_indices, 'Q_out'].values
    I_out_actual = df.loc[selected_indices, 'I_out_actual'].values
    Q_out_actual = df.loc[selected_indices, 'Q_out_actual'].values
    
    # Normalize the data (normalize all together for consistent scaling)
    I_all = np.concatenate([I_out, I_out_actual])
    Q_all = np.concatenate([Q_out, Q_out_actual])
    
    if normalization == 'rms':
        rms = np.sqrt(np.mean(I_all**2 + Q_all**2))
        if rms > 0:
            scale = 1.0 / rms
        else:
            scale = 1.0
    elif normalization == 'max':
        max_mag = np.max(np.sqrt(I_all**2 + Q_all**2))
        if max_mag > 0:
            scale = 1.0 / max_mag
        else:
            scale = 1.0
    else:
        scale = 1.0
    
    # Apply normalization
    I_model_norm = I_out * scale
    Q_model_norm = Q_out * scale
    I_actual_norm = I_out_actual * scale
    Q_actual_norm = Q_out_actual * scale
    
    # Create figure with single plot
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
    
    # Calculate errors for statistics
    error = np.sqrt((I_actual_norm - I_model_norm)**2 + (Q_actual_norm - Q_model_norm)**2)
    mean_error = np.mean(error)
    rms_error = np.sqrt(np.mean(error**2))
    
    # ========== 그래프: Model vs Actual (비교) ==========
    
    # Plot connecting lines for each row (model -> actual)
    for i in range(n_samples):
        ax1.plot([I_model_norm[i], I_actual_norm[i]], 
                [Q_model_norm[i], Q_actual_norm[i]], 
                'b-', alpha=line_alpha, linewidth=0.5, zorder=1)
    
    # Plot constellation points
    ax1.scatter(I_model_norm, Q_model_norm, c='green', label='PA Model Output (I_out/Q_out)',
               alpha=alpha, edgecolors='black', s=point_size, marker='^', linewidths=1.5, zorder=3)
    ax1.scatter(I_actual_norm, Q_actual_norm, c='blue', label='Actual PA Output (I_out_actual/Q_out_actual)',
               alpha=alpha, edgecolors='black', s=point_size, marker='s', linewidths=1.5, zorder=3)
    
    # Set labels and title
    ax1.set_xlabel('I (In-phase)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Q (Quadrature)', fontsize=14, fontweight='bold')
    ax1.set_title('PA Model Output vs. Actual PA Output', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_aspect('equal', adjustable='box')
    
    # Add center lines
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5, zorder=0)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5, zorder=0)
    
    # Statistics for first plot
    stats_text1 = f'Samples: {n_samples}\n'
    stats_text1 += f'Normalization: {normalization}\n'
    stats_text1 += f'\nModel vs Actual Error:\n'
    stats_text1 += f'  Mean: {mean_error:.4f}\n'
    stats_text1 += f'  RMS: {rms_error:.4f}'
    
    ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # CSV 파일과 같은 디렉토리에 PNG 파일로 저장
    csv_path_obj = Path(csv_path)
    output_path = csv_path_obj.parent / "constellation_pa.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()
    
    return fig, output_path


# 실행 코드
if __name__ == '__main__':
    # CSV 경로 확인
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        print(f"Error: CSV file not found: {csv_path}")
        exit(1)
    
    try:
        # 플롯 생성 및 저장
        fig, output_path = plot_constellation_from_csv(
            csv_path,
            n_samples=64,
            seed=None,
            alpha=0.7,
            normalization='rms',
            point_size=50,
            line_alpha=0.3
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

