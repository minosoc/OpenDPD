__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import os
import torch
import models as model
from project import Project
from utils.util import count_net_params
import sys
sys.path.append('../..')
from quant import get_quant_model

def main(proj: Project):
    ###########################################################################################################
    # Initialization
    ###########################################################################################################
    # Set Accelerator Device
    proj.set_device()

    # Build Dataloaders
    (train_loader, val_loader, test_loader), input_size = proj.build_dataloaders()

    ###########################################################################################################
    # Network Settings
    ###########################################################################################################
    # Instantiate PA Model
    net_pa = model.CoreModel(input_size=input_size,
                             hidden_size=proj.PA_hidden_size,
                             num_layers=proj.PA_num_layers,
                             backbone_type=proj.PA_backbone,
                             window_size=proj.window_size,
                             num_dvr_units=proj.num_dvr_units,
                             n_heads=getattr(proj, 'n_heads', 8),
                             d_ff=getattr(proj, 'd_ff', None),
                             dropout=getattr(proj, 'dropout', 0.1))
    n_net_pa_params = count_net_params(net_pa)
    print("::: Number of PA Model Parameters: ", n_net_pa_params)
    pa_model_id = proj.gen_pa_model_id(n_net_pa_params)

    # Load Pretrained PA Model
    pa_model_dir = os.path.join('save', proj.dataset_name, 'train_pa')
    if hasattr(proj.args, 'version') and proj.args.version:
        pa_model_dir = os.path.join(pa_model_dir, proj.args.version)
    path_pa_model = os.path.join(pa_model_dir, pa_model_id + '.pt')
    state_dict = torch.load(path_pa_model)
    # Remove positional encoding from state_dict if present (it's dynamically generated)
    state_dict = {k: v for k, v in state_dict.items() if 'pos_encoding.pe' not in k}
    net_pa.load_state_dict(state_dict, strict=False)

    # Instantiate DPD Model
    net_dpd = model.CoreModel(input_size=input_size,
                              hidden_size=proj.DPD_hidden_size,
                              num_layers=proj.DPD_num_layers,
                              backbone_type=proj.DPD_backbone,
                              window_size=proj.window_size,
                              num_dvr_units=proj.num_dvr_units,
                              thx=proj.thx,
                              thh=proj.thh,
                              n_heads=getattr(proj, 'n_heads', 8),
                              d_ff=getattr(proj, 'd_ff', None),
                              dropout=getattr(proj, 'dropout', 0.1))
    
    net_dpd = get_quant_model(proj, net_dpd)
    
    print("::: DPD Model: ", net_dpd)    
    n_net_dpd_params = count_net_params(net_dpd)
    print("::: Number of DPD Model Parameters: ", n_net_dpd_params)
    dpd_model_id = proj.gen_dpd_model_id(n_net_dpd_params)

    # Instantiate Cascaded Model
    net_cas = model.CascadedModel(dpd_model=net_dpd, pa_model=net_pa)

    # Freeze PA Model
    net_cas.freeze_pa_model()

    # Move the network to the proper device
    net_cas = net_cas.to(proj.device)

    ###########################################################################################################
    # Logger, Loss and Optimizer Settings
    ###########################################################################################################
    # Build Logger
    proj.build_logger(model_id=dpd_model_id)

    # Select Loss function
    criterion = proj.build_criterion()

    # Create Optimizer and Learning Rate Scheduler
    optimizer, lr_scheduler = proj.build_optimizer(net=net_cas)

    ###########################################################################################################
    # Training
    ###########################################################################################################
    proj.train(net=net_cas,
               criterion=criterion,
               optimizer=optimizer,
               lr_scheduler=lr_scheduler,
               train_loader=train_loader,
               val_loader=val_loader,
               test_loader=test_loader,
               best_model_metric='ACLR_AVG')
