__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import os
import time
import pandas as pd
import torch
import numpy as np
import models as model
from modules.paths import create_folder
from project import Project
from utils.util import count_net_params
from modules.data_collector import load_dataset
from modules.train_funcs import calculate_metrics

import sys
sys.path.append('../..')
from quant import get_quant_model
from quant.utlis import register_activation_hooks

def main(proj: Project):
    ###########################################################################################################
    # Initialization
    ###########################################################################################################
    # Set Accelerator Device
    proj.set_device()

    path_dir_output = os.path.join('./pa_out', proj.args.version+'_'+proj.dataset_name)
    if not os.path.exists(path_dir_output):
        os.makedirs(path_dir_output)

    total_start_time = time.time()
    # Load Dataset
    _, _, _, _, X_test, y_test = load_dataset(dataset_name=proj.dataset_name)

    ###########################################################################################################
    # Network Settings
    ###########################################################################################################
    # Instantiate PA Model
    net_pa = model.CoreModel(input_size=2,
                            backbone_type=proj.args.PA_backbone,
                            hidden_size=proj.args.PA_hidden_size,
                            num_layers=proj.args.PA_num_layers,
                            window_size=proj.args.window_size,
                            num_dvr_units=proj.args.num_dvr_units,
                            d_model=proj.args.d_model,
                            n_heads=proj.args.n_heads,
                            d_ff=proj.args.d_ff,
                            dropout_ff=proj.args.dropout_ff,
                            dropout_attn=proj.args.dropout_attn)
    n_net_pa_params = count_net_params(net_pa)
    print("::: Number of PA Model Parameters: ", n_net_pa_params)
    pa_model_id = proj.gen_pa_model_id(n_net_pa_params)

    path_pa_model = os.path.join(proj.path_dir_save, pa_model_id + '.pt')
    # Load Pretrained PA Model (if exists)
    if os.path.exists(path_pa_model):
        print(f"::: Loading Pretrained PA Model: {path_pa_model}")
        state_dict = torch.load(path_pa_model)
        # Remove positional encoding from state_dict if present (it's dynamically generated)
        state_dict = {k: v for k, v in state_dict.items() if 'pos_encoding.pe' not in k}
        net_pa.load_state_dict(state_dict, strict=False)
    else:
        print(f"::: Error: No Pretrained PA Model found: {path_pa_model}")
        raise ValueError(f"::: Error: No Pretrained PA Model found: {path_pa_model} (Please train PA Model first)")
    
    # Move network to device
    net_pa = net_pa.to(proj.device).eval()
    
    ###########################################################################################################
    # Run PA
    ###########################################################################################################
    print("\n::: Running PA inference...")
    # start_time = time.time()
    
    with torch.no_grad():
        # Prepare input (move to device once)
        pa_in = torch.from_numpy(X_test).unsqueeze(dim=0).float().to(proj.device)
        
        # PA forward pass
        print("::: Evaluating PA Performance...")
        pa_start = time.time()
        pa_out = net_pa(pa_in)
        pa_time = time.time() - pa_start
        pa_out = pa_out.squeeze(0).cpu().numpy()
        print(f"::: Inference time - PA: {pa_time:.3f}s")
        
        # Split data into segments for metric calculation (same as IQSegmentDataset)
        # Metrics expect shape (N_segments, nperseg, 2) instead of (N, 2)
        nperseg = getattr(proj.spec, 'nperseg', 2560)
        def split_segments(sequence):
            num_samples = len(sequence)
            segments = []
            for i in range(0, num_samples, nperseg):
                segment = sequence[i:i + nperseg]
                segment = np.asarray(segment)
                if segment.shape[0] < nperseg:
                    padding_shape = (nperseg - segment.shape[0], segment.shape[1])
                    segment = np.vstack((segment, np.zeros(padding_shape, dtype=segment.dtype)))
                segments.append(segment)
            return np.array(segments)
        
        pa_out_segments = split_segments(pa_out)
        # pa_out_segments = split_segments(y_test)
        # y_test = y_test * 1.01
        pa_out_actual_segments = split_segments(y_test)
        
        # Calculate metrics vs actual output (PA's actual output)
        stat = {}
        stat = calculate_metrics(proj.spec, stat, pa_out_segments, pa_out_actual_segments)
        
        # Save metrics to CSV
        metrics_result = 'metrics_' + pa_model_id
        metrics_csv_path = os.path.join(path_dir_output, metrics_result + '.csv')
        metrics_df = pd.DataFrame([stat])
        metrics_df.to_csv(metrics_csv_path, index=False, float_format='%.8f')
        print(f"  - Metrics saved: {metrics_csv_path}")
        
        print("\n" + "="*80)
        print("PA Performance Metrics:")
        print("="*80)
        print("  [vs Actual Output (PA's Actual Output)]")
        print(f"    ACLR_AVG: {stat['ACLR_AVG']:.{proj.args.log_precision}f} dB")
        print(f"    ACLR_L:   {stat['ACLR_L']:.{proj.args.log_precision}f} dB")
        print(f"    ACLR_R:   {stat['ACLR_R']:.{proj.args.log_precision}f} dB")
        print(f"    EVM:      {stat['EVM']:.{proj.args.log_precision}f} dB")
        print(f"    NMSE:     {stat['NMSE']:.{proj.args.log_precision}f} dB")
        print("="*80 + "\n")

    ###########################################################################################################
    # Export Results
    ###########################################################################################################
    print("\n::: Saving outputs...")
    
    # Save combined output: I_in, Q_in, I_out, Q_out, I_out_actual, Q_out_actual, I_out_ideal, Q_out_ideal
    path_file_output = os.path.join(path_dir_output, pa_model_id + '.csv')
    # Round all values to 8 decimal places for consistent formatting
    df = pd.DataFrame({
        'I_in': np.round(X_test[:, 0], 8), 
        'Q_in': np.round(X_test[:, 1], 8), 
        'I_out': np.round(pa_out[:, 0], 8), 
        'Q_out': np.round(pa_out[:, 1], 8),
        'I_out_actual': np.round(y_test[:, 0], 8),
        'Q_out_actual': np.round(y_test[:, 1], 8)
    })
    df.to_csv(path_file_output, index=False, float_format='%.8f')
    print("  - Combined outputs: ", path_file_output)
    print("::: All outputs saved successfully.")
    total_time = time.time() - total_start_time
    print(f"::: Total time: {total_time:.3f}s")