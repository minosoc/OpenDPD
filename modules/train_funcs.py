__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import numpy as np
import types
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import metrics
from typing import Dict, Any, Callable
import argparse


def net_train(log: Dict[str, Any],
                net: nn.Module,
                criterion: Callable,
                optimizer: Optimizer,
                dataloader: DataLoader,
                grad_clip_val: float,
                device: torch.device):
    # Set Network to Training Mode
    net = net.train()
    # Statistics
    losses = []
    # Iterate through batches
    for features, targets in tqdm(dataloader):
        # Move features and targets to the proper device
        features = features.to(device)
        targets = targets.to(device)
        # Initialize all gradients to zero
        optimizer.zero_grad()
        # Forward Propagation
        out = net(features)
        # Calculate the Loss Function
        loss = criterion(out, targets)
        # Backward propagation
        loss.backward()
        # Gradient clipping
        if grad_clip_val != 0:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val)
        # Update parameters
        optimizer.step()
        # Detach loss from the graph indicating the end of forward propagation
        loss.detach()
        # Get losses
        losses.append(loss.item())
    # Average loss
    loss = np.mean(losses)
    # Save Statistics
    log['loss'] = loss
    # End of Training Epoch
    return net


def net_eval(log: Dict,
            net: nn.Module,
            criterion: Callable,
            dataloader: DataLoader,
            device: torch.device):
    net = net.eval()
    with torch.no_grad():
        # Statistics
        losses = []
        prediction = []
        ground_truth = []
        # Batch Iteration
        for features, targets in tqdm(dataloader):
            # Move features and targets to the proper device
            features = features.to(device)
            targets = targets.to(device)
            # Forward Propagation
            out = net(features)
            # Calculate loss function
            loss = criterion(out, targets)
            # Collect prediction and ground truth for metric calculation
            prediction.append(out.cpu())
            ground_truth.append(targets.cpu())
            # Collect losses to calculate the average loss per epoch
            losses.append(loss.item())
    # Average loss
    loss = np.mean(losses)
    # Prediction and Ground Truth
    prediction = torch.cat(prediction, dim=0).numpy()
    ground_truth = torch.cat(ground_truth, dim=0).numpy()
    # Save Statistics
    log['loss'] = loss
    # End of Evaluation Epoch
    return net, prediction, ground_truth


def calculate_metrics(  spec: types.SimpleNamespace,
                        log: Dict[str, Any],
                        prediction: np.ndarray,
                        ground_truth: np.ndarray):
    log['NMSE'] = metrics.NMSE(prediction, ground_truth)
    log['EVM'] = metrics.EVM(prediction, ground_truth, sample_rate=spec.input_signal_fs, bw_main_ch=spec.bw_main_ch, n_sub_ch=spec.n_sub_ch, nperseg=spec.nperseg)
    ACLR_L = []
    ACLR_R = []
    ACLR_left, ACLR_right = metrics.ACLR(prediction, sample_rate=spec.input_signal_fs, bw_main_ch=spec.bw_main_ch, n_sub_ch=spec.n_sub_ch, nperseg=spec.nperseg)
    ACLR_L.append(ACLR_left)
    ACLR_R.append(ACLR_right)
    log['ACLR_L'] = np.mean(ACLR_L)
    log['ACLR_R'] = np.mean(ACLR_R)
    log['ACLR_AVG'] = (log['ACLR_L'] + log['ACLR_R']) / 2
    return log
