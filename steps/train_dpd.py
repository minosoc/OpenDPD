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
    (train_loader, val_loader), input_size = proj.build_dataloaders()

    ###########################################################################################################
    # Network Settings
    ###########################################################################################################
    # Instantiate PA Model
    net_pa = model.CoreModel(input_size=input_size,
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
    if os.path.exists(path_pa_model):
        print(f"::: Loading Pretrained PA Model: {path_pa_model}")
        state_dict = torch.load(path_pa_model)
        # Remove positional encoding from state_dict if present (it's dynamically generated)
        state_dict = {k: v for k, v in state_dict.items() if 'pos_encoding.pe' not in k}
        net_pa.load_state_dict(state_dict, strict=False)
    else:
        print(f"::: Error: No Pretrained PA Model found: {path_pa_model}")
        raise ValueError(f"::: Error: No Pretrained PA Model found: {path_pa_model} (Please train PA Model first)")

    # Instantiate DPD Model
    net_dpd = model.CoreModel(input_size=input_size,
                            backbone_type=proj.args.DPD_backbone,
                            hidden_size=proj.args.DPD_hidden_size,
                            num_layers=proj.args.DPD_num_layers,
                            window_size=proj.args.window_size,
                            num_dvr_units=proj.args.num_dvr_units,
                            d_model=proj.args.d_model,
                            n_heads=proj.args.n_heads,
                            d_ff=proj.args.d_ff,
                            dropout_ff=proj.args.dropout_ff,
                            dropout_attn=proj.args.dropout_attn,
                            thx=proj.args.thx,
                            thh=proj.args.thh)
    
    net_dpd = get_quant_model(proj, net_dpd)
    
    # print("::: DPD Model: ", net_dpd)
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
    proj.train( net=net_cas,
                criterion=criterion,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                best_model_metric='ACLR_AVG')
