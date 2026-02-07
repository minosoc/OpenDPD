#!/usr/bin/env python3
"""
Script to normalize I/Q dataset and save normalization parameters
This script normalizes the dataset and saves the original statistics for later denormalization
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import json

def normalize_dataset(input_dir, output_dir, normalization_method='max_magnitude'):
    """
    Normalize a dataset and save normalization parameters
    
    Args:
        input_dir: Path to input dataset directory (with unnormalized CSV files)
        output_dir: Path to output dataset directory
        normalization_method: 'max_magnitude' or 'min_max'
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load spec.json if exists
    spec_path = input_path / 'spec.json'
    spec = {}
    if spec_path.exists():
        with open(spec_path, 'r') as f:
            spec = json.load(f)
    
    normalization_params = {}
    
    # Check if it's split CSV format or single CSV format
    if (input_path / 'train_input.csv').exists():
        # Split CSV format - calculate statistics from all splits
        print("Processing split CSV format...")
        print("Calculating normalization parameters from all data...")
        
        # Load all data to calculate global statistics
        all_input = []
        all_output = []
        for split in ['train', 'val', 'test']:
            input_file = input_path / f'{split}_input.csv'
            output_file = input_path / f'{split}_output.csv'
            if input_file.exists() and output_file.exists():
                all_input.append(pd.read_csv(input_file))
                all_output.append(pd.read_csv(output_file))
        
        if not all_input:
            raise FileNotFoundError(f"No input CSV files found in {input_path}")
        
        df_input_all = pd.concat(all_input, ignore_index=True)
        df_output_all = pd.concat(all_output, ignore_index=True)
        
        # Calculate normalization parameters
        if normalization_method == 'max_magnitude':
            input_mag = np.sqrt(df_input_all['I']**2 + df_input_all['Q']**2)
            output_mag = np.sqrt(df_output_all['I']**2 + df_output_all['Q']**2)
            
            input_max_mag = input_mag.max()
            output_max_mag = output_mag.max()
            
            normalization_params = {
                'method': 'max_magnitude',
                'input_max_magnitude': float(input_max_mag),
                'output_max_magnitude': float(output_max_mag)
            }
            
            print(f"  Input max magnitude: {input_max_mag:.6f}")
            print(f"  Output max magnitude: {output_max_mag:.6f}")
            
        elif normalization_method == 'min_max':
            input_min = min(df_input_all['I'].min(), df_input_all['Q'].min())
            input_max = max(df_input_all['I'].max(), df_input_all['Q'].max())
            output_min = min(df_output_all['I'].min(), df_output_all['Q'].min())
            output_max = max(df_output_all['I'].max(), df_output_all['Q'].max())
            
            normalization_params = {
                'method': 'min_max',
                'input_min': float(input_min),
                'input_max': float(input_max),
                'output_min': float(output_min),
                'output_max': float(output_max)
            }
            
            print(f"  Input range: [{input_min:.6f}, {input_max:.6f}]")
            print(f"  Output range: [{output_min:.6f}, {output_max:.6f}]")
        
        # Normalize and save each split
        print("\nNormalizing and saving splits...")
        for split in ['train', 'val', 'test']:
            input_file = input_path / f'{split}_input.csv'
            output_file = input_path / f'{split}_output.csv'
            
            if input_file.exists() and output_file.exists():
                df_input = pd.read_csv(input_file)
                df_output = pd.read_csv(output_file)
                
                # Normalize
                if normalization_method == 'max_magnitude':
                    df_input_norm = df_input / input_max_mag
                    df_output_norm = df_output / output_max_mag
                elif normalization_method == 'min_max':
                    input_range = input_max - input_min
                    output_range = output_max - output_min
                    df_input_norm = (df_input - input_min) / input_range
                    df_output_norm = (df_output - output_min) / output_range
                
                # Save normalized data
                df_input_norm.to_csv(output_path / f'{split}_input.csv', index=False)
                df_output_norm.to_csv(output_path / f'{split}_output.csv', index=False)
                
                print(f"  - {split}: normalized and saved")
                in_mag_norm = np.sqrt(df_input_norm['I']**2 + df_input_norm['Q']**2)
                out_mag_norm = np.sqrt(df_output_norm['I']**2 + df_output_norm['Q']**2)
                print(f"    Input mag range: [{in_mag_norm.min():.6f}, {in_mag_norm.max():.6f}]")
                print(f"    Output mag range: [{out_mag_norm.min():.6f}, {out_mag_norm.max():.6f}]")
    
    else:
        # Single CSV format
        csv_file = input_path / 'data.csv'
        if not csv_file.exists():
            raise FileNotFoundError(f"Neither split CSV files nor data.csv found in {input_path}")
        
        print("Processing single CSV format...")
        df = pd.read_csv(csv_file)
        
        # Calculate normalization parameters
        if normalization_method == 'max_magnitude':
            input_mag = np.sqrt(df['I_in']**2 + df['Q_in']**2)
            output_mag = np.sqrt(df['I_out']**2 + df['Q_out']**2)
            
            input_max_mag = input_mag.max()
            output_max_mag = output_mag.max()
            
            normalization_params = {
                'method': 'max_magnitude',
                'input_max_magnitude': float(input_max_mag),
                'output_max_magnitude': float(output_max_mag)
            }
            
            print(f"  Input max magnitude: {input_max_mag:.6f}")
            print(f"  Output max magnitude: {output_max_mag:.6f}")
            
            # Normalize
            df_norm = df.copy()
            df_norm['I_in'] = df['I_in'] / input_max_mag
            df_norm['Q_in'] = df['Q_in'] / input_max_mag
            df_norm['I_out'] = df['I_out'] / output_max_mag
            df_norm['Q_out'] = df['Q_out'] / output_max_mag
            
        elif normalization_method == 'min_max':
            input_min = min(df['I_in'].min(), df['Q_in'].min())
            input_max = max(df['I_in'].max(), df['Q_in'].max())
            output_min = min(df['I_out'].min(), df['Q_out'].min())
            output_max = max(df['I_out'].max(), df['Q_out'].max())
            
            normalization_params = {
                'method': 'min_max',
                'input_min': float(input_min),
                'input_max': float(input_max),
                'output_min': float(output_min),
                'output_max': float(output_max)
            }
            
            print(f"  Input range: [{input_min:.6f}, {input_max:.6f}]")
            print(f"  Output range: [{output_min:.6f}, {output_max:.6f}]")
            
            # Normalize
            input_range = input_max - input_min
            output_range = output_max - output_min
            df_norm = df.copy()
            df_norm['I_in'] = (df['I_in'] - input_min) / input_range
            df_norm['Q_in'] = (df['Q_in'] - input_min) / input_range
            df_norm['I_out'] = (df['I_out'] - output_min) / output_range
            df_norm['Q_out'] = (df['Q_out'] - output_min) / output_range
        
        # Save normalized data
        df_norm.to_csv(output_path / 'data.csv', index=False)
        
        print(f"  - Normalized data.csv saved")
        in_mag_norm = np.sqrt(df_norm['I_in']**2 + df_norm['Q_in']**2)
        out_mag_norm = np.sqrt(df_norm['I_out']**2 + df_norm['Q_out']**2)
        print(f"    Input mag range: [{in_mag_norm.min():.6f}, {in_mag_norm.max():.6f}]")
        print(f"    Output mag range: [{out_mag_norm.min():.6f}, {out_mag_norm.max():.6f}]")
    
    # Save spec.json with normalization parameters
    spec['normalization_params'] = normalization_params
    spec['is_normalized'] = True
    
    with open(output_path / 'spec.json', 'w') as f:
        json.dump(spec, f, indent=4)
    print(f"\nspec.json with normalization parameters saved to: {output_path / 'spec.json'}")
    
    print(f"\nNormalized dataset saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Normalize I/Q dataset and save normalization parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Max magnitude normalization (recommended for I/Q data):
  python normalize_dataset.py datasets/APA_200MHz_unnormalized datasets/APA_200MHz \\
      --method max_magnitude
  
  # Min-Max normalization:
  python normalize_dataset.py datasets/APA_200MHz_unnormalized datasets/APA_200MHz \\
      --method min_max
        """
    )
    parser.add_argument('input_dir', type=str, help='Input dataset directory (unnormalized)')
    parser.add_argument('output_dir', type=str, help='Output dataset directory (normalized)')
    parser.add_argument('--method', type=str, default='max_magnitude',
                       choices=['max_magnitude', 'min_max'],
                       help='Normalization method (default: max_magnitude)')
    
    args = parser.parse_args()
    normalize_dataset(args.input_dir, args.output_dir, args.method)

