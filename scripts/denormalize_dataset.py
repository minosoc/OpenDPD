#!/usr/bin/env python3
"""
Script to denormalize I/Q dataset
This script reads normalized CSV data and converts it back to unnormalized scale
using the original data statistics (min, max, or scale factors)
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import json

def denormalize_dataset(input_dir, output_dir, 
                       input_max_mag=None, output_max_mag=None,
                       input_min=None, input_max=None,
                       output_min=None, output_max=None,
                       normalization_method='max_magnitude'):
    """
    Denormalize a dataset using original data statistics
    
    Args:
        input_dir: Path to input dataset directory (with CSV files)
        output_dir: Path to output dataset directory
        input_max_mag: Original maximum magnitude of input data (for magnitude normalization)
        output_max_mag: Original maximum magnitude of output data (for magnitude normalization)
        input_min: Original minimum value of input data (for min-max normalization)
        input_max: Original maximum value of input data (for min-max normalization)
        output_min: Original minimum value of output data (for min-max normalization)
        output_max: Original maximum value of output data (for min-max normalization)
        normalization_method: 'max_magnitude', 'min_max', or 'global_scale'
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load normalization parameters from spec.json if exists
    spec_path = input_path / 'spec.json'
    if spec_path.exists():
        with open(spec_path, 'r') as f:
            spec = json.load(f)
        
        # Load normalization parameters from spec.json
        norm_params = spec.get('normalization_params', {})
        if norm_params:
            input_max_mag = norm_params.get('input_max_magnitude', input_max_mag)
            output_max_mag = norm_params.get('output_max_magnitude', output_max_mag)
            input_min = norm_params.get('input_min', input_min)
            input_max = norm_params.get('input_max', input_max)
            output_min = norm_params.get('output_min', output_min)
            output_max = norm_params.get('output_max', output_max)
            normalization_method = norm_params.get('method', normalization_method)
            print(f"Loaded normalization parameters from spec.json")
    
    if normalization_method == 'max_magnitude':
        if input_max_mag is None or output_max_mag is None:
            raise ValueError("For max_magnitude normalization, input_max_mag and output_max_mag must be provided")
        print(f"Using max magnitude denormalization:")
        print(f"  Input max magnitude: {input_max_mag}")
        print(f"  Output max magnitude: {output_max_mag}")
    elif normalization_method == 'min_max':
        if any(x is None for x in [input_min, input_max, output_min, output_max]):
            raise ValueError("For min_max normalization, all min/max values must be provided")
        print(f"Using min-max denormalization:")
        print(f"  Input range: [{input_min}, {input_max}]")
        print(f"  Output range: [{output_min}, {output_max}]")
    else:
        raise ValueError(f"Unknown normalization method: {normalization_method}")
    
    # Check if it's split CSV format or single CSV format
    if (input_path / 'train_input.csv').exists():
        # Split CSV format
        print("\nProcessing split CSV format...")
        
        for split in ['train', 'val', 'test']:
            input_file = input_path / f'{split}_input.csv'
            output_file = input_path / f'{split}_output.csv'
            
            if input_file.exists() and output_file.exists():
                # Load normalized data
                df_input = pd.read_csv(input_file)
                df_output = pd.read_csv(output_file)
                
                # Denormalize based on method
                if normalization_method == 'max_magnitude':
                    # Calculate magnitude for each sample
                    input_mag = np.sqrt(df_input['I']**2 + df_input['Q']**2)
                    output_mag = np.sqrt(df_output['I']**2 + df_output['Q']**2)
                    
                    # Avoid division by zero
                    input_mag = np.where(input_mag < 1e-10, 1e-10, input_mag)
                    output_mag = np.where(output_mag < 1e-10, 1e-10, output_mag)
                    
                    # Denormalize: multiply by original max magnitude
                    df_input_denorm = df_input.copy()
                    df_input_denorm['I'] = df_input['I'] * input_max_mag
                    df_input_denorm['Q'] = df_input['Q'] * input_max_mag
                    
                    df_output_denorm = df_output.copy()
                    df_output_denorm['I'] = df_output['I'] * output_max_mag
                    df_output_denorm['Q'] = df_output['Q'] * output_max_mag
                    
                elif normalization_method == 'min_max':
                    # Min-Max denormalization: x_denorm = x_norm * (max - min) + min
                    input_range = input_max - input_min
                    output_range = output_max - output_min
                    
                    df_input_denorm = df_input.copy()
                    df_input_denorm['I'] = df_input['I'] * input_range + input_min
                    df_input_denorm['Q'] = df_input['Q'] * input_range + input_min
                    
                    df_output_denorm = df_output.copy()
                    df_output_denorm['I'] = df_output['I'] * output_range + output_min
                    df_output_denorm['Q'] = df_output['Q'] * output_range + output_min
                
                # Save denormalized data
                df_input_denorm.to_csv(output_path / f'{split}_input.csv', index=False)
                df_output_denorm.to_csv(output_path / f'{split}_output.csv', index=False)
                
                print(f"  - {split}: denormalized and saved")
                in_mag_denorm = np.sqrt(df_input_denorm['I']**2 + df_input_denorm['Q']**2)
                out_mag_denorm = np.sqrt(df_output_denorm['I']**2 + df_output_denorm['Q']**2)
                print(f"    Input: I=[{df_input_denorm['I'].min():.6f}, {df_input_denorm['I'].max():.6f}], "
                      f"Q=[{df_input_denorm['Q'].min():.6f}, {df_input_denorm['Q'].max():.6f}], "
                      f"mag=[{in_mag_denorm.min():.6f}, {in_mag_denorm.max():.6f}]")
                print(f"    Output: I=[{df_output_denorm['I'].min():.6f}, {df_output_denorm['I'].max():.6f}], "
                      f"Q=[{df_output_denorm['Q'].min():.6f}, {df_output_denorm['Q'].max():.6f}], "
                      f"mag=[{out_mag_denorm.min():.6f}, {out_mag_denorm.max():.6f}]")
    else:
        # Single CSV format
        csv_file = input_path / 'data.csv'
        if not csv_file.exists():
            raise FileNotFoundError(f"Neither split CSV files nor data.csv found in {input_path}")
        
        print("\nProcessing single CSV format...")
        df = pd.read_csv(csv_file)
        
        # Denormalize based on method
        if normalization_method == 'max_magnitude':
            # Calculate magnitude for each sample
            input_mag = np.sqrt(df['I_in']**2 + df['Q_in']**2)
            output_mag = np.sqrt(df['I_out']**2 + df['Q_out']**2)
            
            # Avoid division by zero
            input_mag = np.where(input_mag < 1e-10, 1e-10, input_mag)
            output_mag = np.where(output_mag < 1e-10, 1e-10, output_mag)
            
            # Denormalize
            df_denorm = df.copy()
            df_denorm['I_in'] = df['I_in'] * input_max_mag
            df_denorm['Q_in'] = df['Q_in'] * input_max_mag
            df_denorm['I_out'] = df['I_out'] * output_max_mag
            df_denorm['Q_out'] = df['Q_out'] * output_max_mag
            
        elif normalization_method == 'min_max':
            input_range = input_max - input_min
            output_range = output_max - output_min
            
            df_denorm = df.copy()
            df_denorm['I_in'] = df['I_in'] * input_range + input_min
            df_denorm['Q_in'] = df['Q_in'] * input_range + input_min
            df_denorm['I_out'] = df['I_out'] * output_range + output_min
            df_denorm['Q_out'] = df['Q_out'] * output_range + output_min
        
        # Save denormalized data
        df_denorm.to_csv(output_path / 'data.csv', index=False)
        
        print(f"  - Denormalized data.csv saved")
        in_mag_denorm = np.sqrt(df_denorm['I_in']**2 + df_denorm['Q_in']**2)
        out_mag_denorm = np.sqrt(df_denorm['I_out']**2 + df_denorm['Q_out']**2)
        print(f"    Input: I=[{df_denorm['I_in'].min():.6f}, {df_denorm['I_in'].max():.6f}], "
              f"Q=[{df_denorm['Q_in'].min():.6f}, {df_denorm['Q_in'].max():.6f}], "
              f"mag=[{in_mag_denorm.min():.6f}, {in_mag_denorm.max():.6f}]")
        print(f"    Output: I=[{df_denorm['I_out'].min():.6f}, {df_denorm['I_out'].max():.6f}], "
              f"Q=[{df_denorm['Q_out'].min():.6f}, {df_denorm['Q_out'].max():.6f}], "
              f"mag=[{out_mag_denorm.min():.6f}, {out_mag_denorm.max():.6f}]")
    
    # Copy and update spec.json
    if spec_path.exists():
        with open(spec_path, 'r') as f:
            spec = json.load(f)
    else:
        spec = {}
    
    # Update spec to indicate data is now unnormalized
    spec['normalization_params'] = {
        'method': normalization_method,
        'input_max_magnitude': input_max_mag if normalization_method == 'max_magnitude' else None,
        'output_max_magnitude': output_max_mag if normalization_method == 'max_magnitude' else None,
        'input_min': input_min if normalization_method == 'min_max' else None,
        'input_max': input_max if normalization_method == 'min_max' else None,
        'output_min': output_min if normalization_method == 'min_max' else None,
        'output_max': output_max if normalization_method == 'min_max' else None,
    }
    spec['is_normalized'] = False
    
    with open(output_path / 'spec.json', 'w') as f:
        json.dump(spec, f, indent=4)
    print(f"\nspec.json updated and saved to output directory")
    
    print(f"\nDenormalized dataset saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Denormalize I/Q dataset using original data statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Max magnitude normalization (most common for I/Q data):
  python denormalize_dataset.py datasets/APA_200MHz datasets/APA_200MHz_unnormalized \\
      --method max_magnitude --input_max_mag 10.0 --output_max_mag 12.0
  
  # Min-Max normalization:
  python denormalize_dataset.py datasets/APA_200MHz datasets/APA_200MHz_unnormalized \\
      --method min_max --input_min -10.0 --input_max 10.0 --output_min -12.0 --output_max 12.0
  
  # Load parameters from spec.json:
  python denormalize_dataset.py datasets/APA_200MHz datasets/APA_200MHz_unnormalized
        """
    )
    parser.add_argument('input_dir', type=str, help='Input dataset directory')
    parser.add_argument('output_dir', type=str, help='Output dataset directory')
    parser.add_argument('--method', type=str, default='max_magnitude',
                       choices=['max_magnitude', 'min_max'],
                       help='Normalization method (default: max_magnitude)')
    
    # Max magnitude parameters
    parser.add_argument('--input_max_mag', type=float, default=None,
                       help='Original maximum magnitude of input data')
    parser.add_argument('--output_max_mag', type=float, default=None,
                       help='Original maximum magnitude of output data')
    
    # Min-Max parameters
    parser.add_argument('--input_min', type=float, default=None,
                       help='Original minimum value of input data (for min_max method)')
    parser.add_argument('--input_max', type=float, default=None,
                       help='Original maximum value of input data (for min_max method)')
    parser.add_argument('--output_min', type=float, default=None,
                       help='Original minimum value of output data (for min_max method)')
    parser.add_argument('--output_max', type=float, default=None,
                       help='Original maximum value of output data (for min_max method)')
    
    args = parser.parse_args()
    denormalize_dataset(
        args.input_dir, args.output_dir,
        input_max_mag=args.input_max_mag,
        output_max_mag=args.output_max_mag,
        input_min=args.input_min,
        input_max=args.input_max,
        output_min=args.output_min,
        output_max=args.output_max,
        normalization_method=args.method
    )

