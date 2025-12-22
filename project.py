__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import json
import os
import random as rnd
import time
import types
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Callable
from torch import optim
from torch.utils.data import DataLoader
from arguments import get_arguments
from modules.paths import create_folder, gen_log_stat, gen_dir_paths, gen_file_paths
from modules.train_funcs import net_train, net_eval, calculate_metrics
from utils import util
from modules.loggers import PandasLogger
from utils.util import set_target_gain


def dict_to_namespace(d):
    """Recursively convert a dictionary to SimpleNamespace"""
    if isinstance(d, dict):
        return types.SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


class Project:
    def __init__(self):
        ###########################################################################################################
        # Initialization
        ###########################################################################################################
        # Dictionary for Statistics Log
        self.log_all = {}
        self.log_train = {}
        self.log_val = {}
        self.log_test = {}

        # Load Hyperparameters
        self.args = get_arguments()
        self.hparams = vars(self.args)
        for k, v in self.hparams.items():
            setattr(self, k, v)

        # Load Specifications
        self.load_spec()

        # Hardware Info
        self.num_cpu_threads = os.cpu_count()

        # Configure Reproducibility
        self.reproducible()

        ###########################################################################################################
        #  Model ID, Paths of folders and log files and Logger
        ###########################################################################################################
        # Create Folders
        dir_paths = gen_dir_paths(self.args)
        self.path_dir_save, self.path_dir_log, self.path_dir_terminal_log = dir_paths
        create_folder([self.path_dir_save, self.path_dir_log, self.path_dir_terminal_log])

    def gen_pa_model_id(self, n_net_params):
        dict_pa = { 'S': f"{self.seed}",
                    'M': self.PA_backbone.upper(),
                    }
        if 'transformer' in self.PA_backbone:
            dict_pa['D'] = f"{self.d_model:d}"
            dict_pa['H'] = f"{self.n_heads:d}"
            dict_pa['L'] = f"{self.PA_num_layers:d}"
            dict_pa['P'] = f"{n_net_params:d}"
        else:
            dict_pa['H'] = f"{self.PA_hidden_size:d}"
            dict_pa['L'] = f"{self.PA_num_layers:d}"
            dict_pa['P'] = f"{n_net_params:d}"
        dict_pamodel_id = dict(list(dict_pa.items()))

        # PA Model ID
        list_pamodel_id = []
        for item in list(dict_pamodel_id.items()):
            list_pamodel_id += list(item)
        pa_model_id = '_'.join(list_pamodel_id)
        pa_model_id = 'PA_' + pa_model_id
        return pa_model_id

    def gen_dpd_model_id(self, n_net_params):
        dict_dpd = {'S': f"{self.seed}",
                    'M': self.DPD_backbone.upper(),
                    'H': f"{self.gen_dpd_model_id:d}"
                    }
        if 'transformer' in self.DPD_backbone:
            dict_dpd['D'] = f"{self.d_model:d}"
            dict_dpd['H'] = f"{self.n_heads:d}"
            dict_dpd['L'] = f"{self.DPD_num_layers:d}"
            dict_dpd['P'] = f"{n_net_params:d}"
        else:
            dict_dpd['H'] = f"{self.DPD_hidden_size:d}"
            dict_dpd['L'] = f"{self.DPD_num_layers:d}"
            dict_dpd['P'] = f"{n_net_params:d}"
        if 'delta' in self.DPD_backbone:
            dict_dpd['THX'] = f"{self.thx:.3f}"
            dict_dpd['THH'] = f"{self.thh:.3f}"
        dict_dpdmodel_id = dict(list(dict_dpd.items()))

        # DPD Model ID
        list_dpdmodel_id = []
        for item in list(dict_dpdmodel_id.items()):
            list_dpdmodel_id += list(item)
        dpd_model_id = '_'.join(list_dpdmodel_id)
        dpd_model_id = 'DPD_' + dpd_model_id
        return dpd_model_id

    def build_logger(self, model_id: str):
        # Get Save and Log Paths
        file_paths = gen_file_paths(self.path_dir_save, self.path_dir_log, self.path_dir_terminal_log, model_id, self.args.step)
        self.path_file_save_best, self.path_file_log_hist, self.path_file_log_best, self.path_file_terminal_log = file_paths
        print("::: Best Model Save Path: ", self.path_file_save_best)
        print("::: Log-History     Path: ", self.path_file_log_hist)
        print("::: Log-Best        Path: ", self.path_file_log_best)
        print("::: Terminal Log    Path: ", self.path_file_terminal_log)
        
        # Instantiate Logger for Recording Training Statistics
        self.logger = PandasLogger(path_file_save_best=self.path_file_save_best,
                                    path_file_log_hist=self.path_file_log_hist,
                                    path_file_log_best=self.path_file_log_best,
                                    path_file_terminal_log=self.path_file_terminal_log,
                                    precision=self.log_precision)

    def reproducible(self):
        rnd.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # torch.autograd.set_detect_anomaly(True)

        if self.re_level == 'soft':
            torch.use_deterministic_algorithms(mode=False)
            torch.backends.cudnn.benchmark = True
        else:  # re_level == 'hard'
            torch.use_deterministic_algorithms(mode=True)
            torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        print("::: Are Deterministic Algorithms Enabled: ", torch.are_deterministic_algorithms_enabled())
        print("--------------------------------------------------------------------")

    def load_spec(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Get relative path to the spec file
        spec = {}
        if hasattr(self, 'dataset_path') and self.dataset_path:
            # Custom dataset path
            dataset_path = os.path.abspath(self.dataset_path)
            self.dataset_path = dataset_path
            if os.path.isfile(dataset_path) and dataset_path.endswith('.csv'):
                # Single CSV file - create minimal spec
                spec = {
                    'dataset_format': 'single_csv',
                    'split_ratios': {
                        'train': 0.6,
                        'val': 0.2,
                        'test': 0.2
                    },
                    'nperseg': 2560  # Default value
                }
                path_spec = None
            elif os.path.isdir(dataset_path):
                path_spec = os.path.join(dataset_path, 'spec.json')
            else:
                raise ValueError(f"Invalid dataset path: {self.dataset_path}")
        else:
            # Standard dataset name
            path_spec = os.path.join(base_dir, 'datasets', self.dataset_name, 'spec.json')

        # Load the spec
        if path_spec and os.path.exists(path_spec):
            with open(path_spec) as config_file:
                spec = json.load(config_file)
        elif path_spec and hasattr(self, 'dataset_path') and self.dataset_path:
            raise FileNotFoundError(f"spec.json not found in dataset path: {self.dataset_path}")
        elif path_spec:
            # No spec file and no dataset_path - this shouldn't happen
            raise FileNotFoundError(f"spec.json not found for dataset: {self.dataset_name}")
        
        if spec:
            self.spec = dict_to_namespace(spec)
            for k, v in spec.items():
                setattr(self, k, v)
                self.hparams[k] = v

    def add_arg(self, key: str, value: Any):
        setattr(self, key, value)
        setattr(self.args, key, value)
        self.hparams[key] = value

    def set_device(self):
        # Find Available GPUs
        if self.accelerator == 'cuda' and torch.cuda.is_available():
            idx_gpu = self.args.devices
            name_gpu = torch.cuda.get_device_name(idx_gpu)
            device = torch.device("cuda:" + str(idx_gpu))
            torch.cuda.set_device(device)
            print("::: Available GPUs: %s" % (torch.cuda.device_count()))
            print("::: Using GPU %s:   %s" % (idx_gpu, name_gpu))
            print("--------------------------------------------------------------------")
        elif self.accelerator == 'mps' and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif self.accelerator == 'cpu':
            device = torch.device("cpu")
            print("::: Available GPUs: None")
            print("--------------------------------------------------------------------")
        else:
            raise ValueError(f"The select device {self.accelerator} is not supported.")
        self.add_arg("device", device)
        return device

    def build_dataloaders(self):
        from modules.data_collector import IQSegmentDataset, IQFrameDataset, load_dataset

        # Load Dataset
        if hasattr(self, 'dataset_path') and self.dataset_path:
            X_train, y_train, X_val, y_val = load_dataset(dataset_path=self.dataset_path)
        else:
            X_train, y_train, X_val, y_val = load_dataset(dataset_name=self.dataset_name)

        # Apply the PA Gain if training DPD
        # For DPD training, target should be ideal linear amplification (input * target_gain)
        # NOT the actual PA output (train_output), because DPD aims to make PA output linear
        # Calculate target_gain using all data (train + val + test) for better estimation
        # X_all = np.concatenate([X_train, X_val, X_test], axis=0)
        # y_all = np.concatenate([y_train, y_val, y_test], axis=0)
        self.target_gain = getattr(self.spec, 'gain', None)
        if self.target_gain is None:
            raise ValueError("'gain' is required in spec.json but not found. Please add 'gain' to your spec.json file.")
        if self.args.step == 'train_dpd':
            y_train = self.target_gain * X_train        # G * x (ideal)
            y_val = self.target_gain * X_val

        # Extract Features
        input_size = X_train.shape[-1]

        # Define PyTorch Datasets
        train_set = IQFrameDataset(X_train, y_train, frame_length=self.frame_length, stride=self.frame_stride)
        val_set = IQSegmentDataset(X_val, y_val, nperseg=self.args.nperseg)

        # Define PyTorch Dataloaders
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size_eval, shuffle=False)

        return (train_loader, val_loader), input_size

    # def build_model(self):
    #     # Load Pretrained Model if Running Retrain
    #     if self.step == 'retrain':
    #         net = self.net_retrain.Model(self)  # Instantiate Retrain Model
    #         if self.path_net_pretrain is None:
    #             print('::: Loading pretrained model: ', self.default_path_net_pretrain)
    #             # net = util.load_model(self, net, self.default_path_net_pretrain)
    #             net.load_pretrain_model(self.default_path_net_pretrain)
    #         else:
    #             print('::: Loading pretrained model: ', self.path_net_pretrain)
    #             net = util.load_model(self, net, self.path_net_pretrain)
    #     else:
    #         net = self.net_pretrain.Model(self)  # Instantiate Pretrain Model

    #     # Cast net to the target device
    #     net.to(self.device)
    #     self.add_arg("net", net)

    #     return net

    def build_criterion(self):
        dict_loss = {'l2': nn.MSELoss(),
                    'l1': nn.L1Loss()
                    }
        loss_func_name = self.loss_type
        try:
            criterion = dict_loss[loss_func_name]
            self.add_arg("criterion", criterion)
            return criterion
        except AttributeError:
            raise AttributeError('Please use a valid loss function. Check argument.py.')

    def build_optimizer(self, net: nn.Module):
        # Optimizer
        if self.opt_type == 'adam':
            optimizer = optim.Adam(net.parameters(), lr=self.lr)
        elif self.opt_type == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=0.9)
        elif self.opt_type == 'rmsprop':
            optimizer = optim.RMSprop(net.parameters(), lr=self.lr)
        elif self.opt_type == 'adamw':
            optimizer = optim.AdamW(net.parameters(), lr=self.lr)
        elif self.opt_type == 'adabound':
            import adabound  # Run pip install adabound (https://github.com/Luolc/AdaBound)
            optimizer = adabound.AdaBound(net.parameters(), lr=self.lr, final_lr=0.1)
        else:
            raise RuntimeError('Please use a valid optimizer.')

        # Learning Rate Scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                            mode='min',
                                                            factor=self.decay_factor,
                                                            patience=self.patience,
                                                            threshold=1e-4,
                                                            min_lr=self.lr_end)
        return optimizer, lr_scheduler

    def train(  self,
                net: nn.Module,
                criterion: Callable,
                optimizer: optim.Optimizer,
                lr_scheduler,
                train_loader: DataLoader,
                val_loader: DataLoader,
                best_model_metric: str) -> None:
        # Timer
        start_time = time.time()
        # Epoch loop
        print("Starting training...")
        for epoch in range(self.args.n_epochs):
            # -----------
            # Train
            # -----------
            net = net_train(log=self.log_train,
                            net=net,
                            criterion=criterion,
                            optimizer=optimizer,
                            dataloader=train_loader,
                            grad_clip_val=self.args.grad_clip_val,
                            device=self.device)

            # -----------
            # Validation
            # -----------
            if self.args.eval_val:
                _, prediction, ground_truth = net_eval(log=self.log_val,
                                                        net=net,
                                                        criterion=criterion,
                                                        dataloader=val_loader,
                                                        device=self.device)
                self.log_val = calculate_metrics(self.spec, self.log_val, prediction, ground_truth)

            ###########################################################################################################
            # Logging & Saving
            ###########################################################################################################

            # Generate Log Dict
            end_time = time.time()
            elapsed_time_minutes = (end_time - start_time) / 60.0
            self.log_all = gen_log_stat(self.args, elapsed_time_minutes, net, optimizer, epoch, self.log_train, self.log_val)

            # Write Log
            self.logger.write_log(self.log_all)

            # Save best model
            best_net = net.dpd_model if self.args.step == 'train_dpd' else net
            self.logger.save_best_model(net=best_net, epoch=epoch, val_stat=self.log_val, metric_name=best_model_metric)

            ###########################################################################################################
            # Learning Rate Schedule
            ###########################################################################################################
            # Schedule at the beginning of retrain
            lr_scheduler_criteria = self.log_val[best_model_metric]
            if self.args.lr_schedule:
                lr_scheduler.step(lr_scheduler_criteria)

        print("Training Completed...")
        print(" ")