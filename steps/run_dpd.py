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
from utils.util import count_net_params, set_target_gain
from modules.data_collector import load_dataset
from modules.train_funcs import calculate_metrics

import sys
sys.path.append('../..')
from quant import get_quant_model
from quant.utlis import register_activation_hooks

def main(proj: Project):
    total_start_time = time.time()
    ###########################################################################################################
    # Initialization
    ###########################################################################################################
    # Set Accelerator Device
    proj.set_device()

    # Load Dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name=proj.dataset_name)
    
    # Calculate target gain for evaluation using all data (train + val + test) for better estimation
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    target_gain = set_target_gain(X_all, y_all)

    # Create DPD Output Folder
    output_dir = 'dpd_out'
    if hasattr(proj.args, 'version') and proj.args.version:
        output_dir = os.path.join('dpd_out', proj.args.version)
    if proj.args.quant and proj.args.quant_dir_label:
        if output_dir == 'dpd_out':
            output_dir = os.path.join('dpd_out', proj.args.quant_dir_label)
        else:
            output_dir = os.path.join(output_dir, proj.args.quant_dir_label)
    create_folder([output_dir])

    ###########################################################################################################
    # Network Settings
    ###########################################################################################################
    # Instantiate PA Model
    net_pa = model.CoreModel(input_size=2,
                             hidden_size=proj.PA_hidden_size,
                             num_layers=proj.PA_num_layers,
                             backbone_type=proj.PA_backbone,
                             num_dvr_units=proj.num_dvr_units)
    n_net_pa_params = count_net_params(net_pa)
    print("::: Number of PA Model Parameters: ", n_net_pa_params)
    pa_model_id = proj.gen_pa_model_id(n_net_pa_params)
    
    # Load Pretrained PA Model
    path_pa_model = os.path.join('save', proj.dataset_name, 'train_pa', pa_model_id + '.pt')
    print("::: Loading PA Model: ", path_pa_model)
    net_pa.load_state_dict(torch.load(path_pa_model, map_location='cpu'))
    
    # Instantiate DPD Model
    net_dpd = model.CoreModel(input_size=2,
                              hidden_size=proj.DPD_hidden_size,
                              num_layers=proj.DPD_num_layers,
                              backbone_type=proj.DPD_backbone)
    
    # Determine DPD model path (pretrained_model is required)
    if not proj.args.pretrained_model:
        raise ValueError("--pretrained_model is required. Please specify the DPD model path.")
    pretrained_model = proj.args.pretrained_model
    print("::: Using specified DPD model path: ", pretrained_model)
    
    # Load DPD Model (quantization handling)
    load_start_time = time.time()
    if proj.args.quant and proj.args.pretrained_model:
        # For quantization models with pretrained_model: create structure first, then load weights
        original_pretrained = proj.args.pretrained_model
        proj.args.pretrained_model = ''  # Prevent get_quant_model from loading
        
        net_dpd = get_quant_model(proj, net_dpd)
        
        if not hasattr(proj, 'quant_env') or proj.quant_env is None:
            raise RuntimeError("Quantization setup failed. Cannot load quantization model.")
        
        proj.args.pretrained_model = original_pretrained  # Restore
        
        print("::: Loading Quantization DPD Model: ", original_pretrained)
        state_dict = torch.load(original_pretrained, map_location='cpu')
        missing_keys, unexpected_keys = net_dpd.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"::: Warning: {len(missing_keys)} missing keys (using defaults)")
        total_load_time = time.time() - load_start_time
        print(f"::: Quantization model loaded successfully (Total: {total_load_time:.3f}s)")
    else:
        # Normal flow: get_quant_model or load directly
        net_dpd = get_quant_model(proj, net_dpd)
        
        print("::: Loading DPD Model: ", proj.args.pretrained_model)
        state_dict = torch.load(proj.args.pretrained_model, map_location='cpu')
        try:
            net_dpd.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            net_dpd.load_state_dict(state_dict, strict=False)
            print("::: Loaded with strict=False (quantization model)")
        
        total_load_time = time.time() - load_start_time
        print(f"::: DPD model loaded successfully (Total: {total_load_time:.3f}s)")
    
    n_net_dpd_params = count_net_params(net_dpd)
    dpd_model_id = proj.gen_dpd_model_id(n_net_dpd_params)
    print("::: Number of DPD Model Parameters: ", n_net_dpd_params)

    # Move networks to device and create cascaded model
    net_pa = net_pa.to(proj.device).eval()
    net_dpd = net_dpd.to(proj.device).eval()
    net_cas = model.CascadedModel(dpd_model=net_dpd, pa_model=net_pa).eval()
    
    # Pre-compute quantization scales for inference optimization
    if proj.args.quant:
        def precompute_quant_scales(model):
            """Pre-compute quantization scales for all quantizers in eval mode"""
            for module in model.modules():
                # Check all possible quantizer attributes
                for attr_name in ['weight_quantizer', 'act_quantizer', 'out_quantizer', 'quantizer']:
                    if hasattr(module, attr_name):
                        quantizer = getattr(module, attr_name)
                        if hasattr(quantizer, 'round_scale2pow2') and hasattr(quantizer, 'scale'):
                            # Force computation of pow2_scale
                            pow2_scale, dec_num = quantizer.round_scale2pow2(quantizer.scale)
                            quantizer.update_params(pow2_scale, dec_num)
        
        # Pre-compute scales after model is loaded and in eval mode
        precompute_quant_scales(net_dpd)
    
    # Optimize for inference (simple optimizations only)
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    torch.backends.cudnn.deterministic = False

    ###########################################################################################################
    # Run DPD
    ###########################################################################################################
    print("\n::: Running DPD inference...")
    start_time = time.time()
    
    with torch.no_grad():
        # Prepare input (move to device once)
        dpd_in = torch.from_numpy(X_test).unsqueeze(dim=0).float().to(proj.device)
        
        # Warm-up pass (for quantization models, first pass can be slow)
        if proj.args.quant:
            _ = net_dpd(dpd_in[:1, :100, :])  # Small warm-up
        
        # DPD forward pass
        dpd_start = time.time()
        dpd_out = net_dpd(dpd_in)
        dpd_time = time.time() - dpd_start
        dpd_out = dpd_out.squeeze(0).cpu().numpy()
        
        # Cascaded forward pass (DPD + PA)
        print("::: Evaluating DPD Performance with PA Model...")
        cascaded_start = time.time()
        cascaded_out = net_cas(dpd_in)
        cascaded_time = time.time() - cascaded_start
        cascaded_out = cascaded_out.squeeze(0).cpu().numpy()
        
        total_time = time.time() - start_time
        print(f"::: Inference time - DPD: {dpd_time:.3f}s, Cascaded: {cascaded_time:.3f}s, Total: {total_time:.3f}s")
        
        # Ground truth: ideal linear amplification (target_gain * input)
        # This is what DPD aims to achieve: make PA output linear amplification of input
        ground_truth_ideal = target_gain * X_test
        
        # Split data into segments for metric calculation (same as IQSegmentDataset)
        # Metrics expect shape (N_segments, nperseg, 2) instead of (N, 2)
        nperseg = proj.args.nperseg
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
        
        cascaded_out_segments = split_segments(cascaded_out)
        ground_truth_ideal_segments = split_segments(ground_truth_ideal)
        
        # Calculate metrics vs ideal linear amplification (DPD's goal)
        stat = {}
        stat = calculate_metrics(proj.args, stat, cascaded_out_segments, ground_truth_ideal_segments)
        
        print("\n" + "="*80)
        print("DPD Performance Metrics:")
        print("="*80)
        print("  [vs Ideal Linear Amplification (DPD's Target)]")
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
    
    # Save combined output: I, Q, I_pa_out, Q_pa_out, I_pa_out_ideal, Q_pa_out_ideal
    path_file_output = os.path.join(output_dir, dpd_model_id + '.csv')
    # Round all values to 8 decimal places for consistent formatting
    df = pd.DataFrame({
        'I': np.round(X_test[:, 0], 8), 
        'Q': np.round(X_test[:, 1], 8), 
        'I_pa_out': np.round(cascaded_out[:, 0], 8), 
        'Q_pa_out': np.round(cascaded_out[:, 1], 8),
        'I_pa_out_ideal': np.round(ground_truth_ideal[:, 0], 8),
        'Q_pa_out_ideal': np.round(ground_truth_ideal[:, 1], 8)
    })
    df.to_csv(path_file_output, index=False, float_format='%.8f')
    print("  - Combined outputs: ", path_file_output)
    print("::: All outputs saved successfully.")
    total_time = time.time() - total_start_time
    print(f"::: Total time: {total_time:.3f}s")