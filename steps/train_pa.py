__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import models as model
from project import Project
from utils.util import count_net_params


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
    # Instantiate Model
    net = model.CoreModel(  input_size=input_size,
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
    n_net_pa_params = count_net_params(net)
    print("::: Number of PA Model Parameters: ", n_net_pa_params)
    pa_model_id = proj.gen_pa_model_id(n_net_pa_params)

    # Move the network to the proper device
    net = net.to(proj.device)

    ###########################################################################################################
    # Logger, Loss and Optimizer Settings
    ###########################################################################################################
    # Build Logger
    proj.build_logger(model_id=pa_model_id)

    # Select Loss function
    criterion = proj.build_criterion()

    # Create Optimizer and Learning Rate Scheduler
    optimizer, lr_scheduler = proj.build_optimizer(net=net)

    ###########################################################################################################
    # Training
    ###########################################################################################################
    proj.train( net=net,
                criterion=criterion,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                best_model_metric='NMSE')
