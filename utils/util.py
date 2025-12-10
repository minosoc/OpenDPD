__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import numpy as np


def count_net_params(net):
    n_param = 0
    for name, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes
    return n_param


def get_amplitude(IQ_signal):
    I = IQ_signal[:, 0]
    Q = IQ_signal[:, 1]
    power = I ** 2 + Q ** 2
    amplitude = np.sqrt(power)
    return amplitude


def set_target_gain(input_IQ, output_IQ, linear_region_percentile=80):
    """Calculate the target gain from the linear region of PA amplification.
    
    PA amplifiers typically exhibit linear behavior at low input levels and 
    non-linear behavior (saturation, compression) at high input levels. This 
    function calculates the gain only from the linear region to get a more 
    accurate target gain for DPD training.
    
    Args:
        input_IQ: Input I-Q signal (N, 2)
        output_IQ: Output I-Q signal (N, 2)
        linear_region_percentile: Percentile threshold for linear region (default: 80)
                                 Samples below this percentile are considered linear
    
    Returns:
        target_gain: Average gain from the linear region
    """
    amp_in = get_amplitude(input_IQ)
    amp_out = get_amplitude(output_IQ)
    
    # Find the threshold for linear region (exclude high-power samples)
    # np.percentile sorts values by magnitude and finds the value at the given percentile
    # For example, percentile=80 means: after sorting by value, 80% of samples are below this threshold
    # This selects the linear region (low input power samples)
    amp_threshold = np.percentile(amp_in, linear_region_percentile)
    
    # Select samples in the linear region (low input power)
    linear_mask = amp_in <= amp_threshold
    
    # Ensure we have enough samples in the linear region
    if np.sum(linear_mask) < len(amp_in) * 0.1:  # At least 10% of samples
        print(f"::: Warning: Linear region too small ({np.sum(linear_mask)} samples). Using all samples.")
        linear_mask = np.ones_like(amp_in, dtype=bool)
    
    # Extract linear region samples
    amp_in_linear = amp_in[linear_mask]
    amp_out_linear = amp_out[linear_mask]
    
    # Avoid division by zero
    amp_in_safe = np.where(amp_in_linear == 0, np.finfo(float).eps, amp_in_linear)
    
    # Calculate gain for each sample in linear region and take the mean
    gains_linear = amp_out_linear / amp_in_safe
    target_gain = np.mean(gains_linear)
    
    print(f"::: Target gain calculated from linear region:")
    print(f"    Linear region: {np.sum(linear_mask)}/{len(amp_in)} samples ({100*np.sum(linear_mask)/len(amp_in):.1f}%)")
    print(f"    Input amplitude threshold: {amp_threshold:.6f} (percentile {linear_region_percentile})")
    print(f"    Target gain: {target_gain:.6f}")
    
    return target_gain
