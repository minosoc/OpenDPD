#!/usr/bin/env python3
"""
Script to plot constellation diagram from CSV file containing I/Q data.

This script reads a CSV file and plots a constellation diagram showing:
- Ideal output (I_out_ideal/Q_out_ideal)
- Actual PA output (I_out_actual/Q_out_actual)
- Cascaded DPD+PA output (I_out/Q_out)

For 64 randomly selected rows, showing how DPD corrects the PA distortion.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import IQ_to_complex


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
    Read CSV file and plot constellation diagram showing DPD correction.
    
    Selects n_samples random rows and plots three points for each:
    - Ideal output (I_out_ideal/Q_out_ideal) - red
    - Actual PA output (I_out_actual/Q_out_actual) - blue  
    - Cascaded DPD+PA output (I_out/Q_out) - green
    
    Parameters:
    - csv_path: Path to CSV file
    - n_samples: Number of rows to randomly select (default: 64)
    - seed: Random seed for reproducibility
    - alpha: Transparency for scatter points
    - normalization: 'rms', 'max', or 'none' for normalization method
    - point_size: Size of scatter points
    - line_alpha: Transparency for connecting lines
    """
    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ['I_out', 'Q_out', 'I_out_actual', 'Q_out_actual', 'I_out_ideal', 'Q_out_ideal']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. "
                        f"Found columns: {df.columns.tolist()}")
    
    # Set random seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Select random rows
    total_rows = len(df)
    n_samples = min(n_samples, total_rows)
    selected_indices = random.sample(range(total_rows), n_samples)
    selected_indices.sort()  # Sort for better visualization
    
    print(f"Selected {n_samples} random rows out of {total_rows} total rows")
    
    # Extract I/Q data for selected rows
    I_out = df.loc[selected_indices, 'I_out'].values
    Q_out = df.loc[selected_indices, 'Q_out'].values
    I_out_actual = df.loc[selected_indices, 'I_out_actual'].values
    Q_out_actual = df.loc[selected_indices, 'Q_out_actual'].values
    I_out_ideal = df.loc[selected_indices, 'I_out_ideal'].values
    Q_out_ideal = df.loc[selected_indices, 'Q_out_ideal'].values
    
    # Normalize the data (normalize all together for consistent scaling)
    # Use ideal output for normalization reference
    I_all = np.concatenate([I_out_ideal, I_out_actual, I_out])
    Q_all = np.concatenate([Q_out_ideal, Q_out_actual, Q_out])
    
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
    I_ideal_norm = I_out_ideal * scale
    Q_ideal_norm = Q_out_ideal * scale
    I_actual_norm = I_out_actual * scale
    Q_actual_norm = Q_out_actual * scale
    I_cascaded_norm = I_out * scale
    Q_cascaded_norm = Q_out * scale
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot connecting lines for each row (ideal -> actual, ideal -> cascaded)
    for i in range(n_samples):
        # Line from ideal to actual
        ax.plot([I_ideal_norm[i], I_actual_norm[i]], 
                [Q_ideal_norm[i], Q_actual_norm[i]], 
                'b-', alpha=line_alpha, linewidth=0.5, zorder=1)
        # Line from ideal to cascaded
        ax.plot([I_ideal_norm[i], I_cascaded_norm[i]], 
                [Q_ideal_norm[i], Q_cascaded_norm[i]], 
                'g-', alpha=line_alpha, linewidth=0.5, zorder=1)
    
    # Plot constellation points
    # Ideal output (reference) - red circles
    ax.scatter(I_ideal_norm, Q_ideal_norm, c='red', label='Ideal Output (I_out_ideal/Q_out_ideal)',
               alpha=alpha, edgecolors='black', s=point_size, marker='o', linewidths=1.5, zorder=3)
    
    # Actual PA output - blue squares
    ax.scatter(I_actual_norm, Q_actual_norm, c='blue', label='Actual PA Output (I_out_actual/Q_out_actual)',
               alpha=alpha, edgecolors='black', s=point_size, marker='s', linewidths=1.5, zorder=3)
    
    # Cascaded DPD+PA output - green triangles
    ax.scatter(I_cascaded_norm, Q_cascaded_norm, c='green', label='Cascaded DPD+PA Output (I_out/Q_out)',
               alpha=alpha, edgecolors='black', s=point_size, marker='^', linewidths=1.5, zorder=3)
    
    # Set labels and title
    ax.set_xlabel('I (In-phase)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Q (Quadrature)', fontsize=14, fontweight='bold')
    ax.set_title('Constellation Diagram: DPD Correction Visualization\n(Ideal → Actual → Cascaded)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Add center lines
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5, zorder=0)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5, zorder=0)
    
    # Calculate and display statistics
    error_actual = np.sqrt((I_actual_norm - I_ideal_norm)**2 + (Q_actual_norm - Q_ideal_norm)**2)
    error_cascaded = np.sqrt((I_cascaded_norm - I_ideal_norm)**2 + (Q_cascaded_norm - Q_ideal_norm)**2)
    
    mean_error_actual = np.mean(error_actual)
    rms_error_actual = np.sqrt(np.mean(error_actual**2))
    mean_error_cascaded = np.mean(error_cascaded)
    rms_error_cascaded = np.sqrt(np.mean(error_cascaded**2))
    improvement = ((mean_error_actual - mean_error_cascaded) / mean_error_actual) * 100 if mean_error_actual > 0 else 0
    
    stats_text = f'Samples shown: {n_samples}\n'
    stats_text += f'Normalization: {normalization}\n'
    stats_text += f'\nActual PA Error:\n'
    stats_text += f'  Mean: {mean_error_actual:.4f}\n'
    stats_text += f'  RMS: {rms_error_actual:.4f}\n'
    stats_text += f'\nCascaded Error:\n'
    stats_text += f'  Mean: {mean_error_cascaded:.4f}\n'
    stats_text += f'  RMS: {rms_error_cascaded:.4f}\n'
    stats_text += f'\nImprovement: {improvement:.1f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description='Plot constellation diagram from CSV file with I/Q data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/plot_constellation_from_csv.py cascaded_out/rev3_APA_200MHz/DPD_S_0_M_TRANSFORMER_ENCODER_D_128_H_8_L_20_P_3966082.csv
  
  python scripts/plot_constellation_from_csv.py data.csv --n-samples 64 --seed 42 --save output.png
        """
    )
    
    parser.add_argument('csv_path', type=str,
                       help='Path to CSV file containing I_out, Q_out, I_out_actual, Q_out_actual, I_out_ideal, Q_out_ideal columns')
    parser.add_argument('--n-samples', type=int, default=64,
                       help='Number of random rows to select and plot (default: 64)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Transparency for scatter points (default: 0.7)')
    parser.add_argument('--normalization', type=str, default='rms',
                       choices=['rms', 'max', 'none'],
                       help='Normalization method: rms (default), max, or none')
    parser.add_argument('--point-size', type=float, default=50.0,
                       help='Size of scatter points (default: 50.0)')
    parser.add_argument('--line-alpha', type=float, default=0.3,
                       help='Transparency for connecting lines (default: 0.3)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save figure to file instead of displaying (e.g., output.png)')
    
    args = parser.parse_args()
    
    # Resolve CSV path
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    try:
        fig, ax = plot_constellation_from_csv(csv_path, 
                                             n_samples=args.n_samples,
                                             seed=args.seed,
                                             alpha=args.alpha,
                                             normalization=args.normalization,
                                             point_size=args.point_size,
                                             line_alpha=args.line_alpha)
        
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {args.save}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

