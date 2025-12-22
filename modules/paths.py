__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import argparse
import os


def gen_log_stat(args: argparse.Namespace, elapsed_time, net, optimizer, epoch, train_stat=None, val_stat=None):
    # Get Epoch & Batch Size
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    # Get current learning rate
    lr_curr = 0
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

    # Get parameter count
    n_param = 0
    for name, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes

    if args.step == 'train_pa':
        backbone = args.PA_backbone
        hidden_size = args.PA_hidden_size
    elif args.step == 'train_dpd':
        backbone = args.DPD_backbone
        hidden_size = args.DPD_hidden_size

    # Create log dictionary
    log_stat = {'EPOCH': epoch,
                'N_EPOCH': n_epochs,
                'TIME:': elapsed_time,
                'LR': lr_curr,
                'BATCH_SIZE': batch_size,
                'N_PARAM': n_param,
                'FRAME_LENGTH': args.frame_length,
                'BACKBONE': backbone,
                'HIDDEN_SIZE': hidden_size,
                }
    
    if 'transformer' in backbone:
        log_stat['D_MODEL'] = args.d_model
        log_stat['N_HEADS'] = args.n_heads
        log_stat['D_FF'] = args.d_ff
        log_stat['NUM_LAYERS'] = args.PA_num_layers
    # Add threshold values to log
    if args.step == 'train_dpd':
        if 'delta' in net.dpd_model.backbone_type:
            log_stat['THX'] = net.dpd_model.backbone.thx
            log_stat['THH'] = net.dpd_model.backbone.thh
            
            # Add sparsity metrics if available
            if 'delta' in net.dpd_model.backbone_type:
                sparsity_metrics = net.dpd_model.backbone.get_temporal_sparsity()
                sparsity_log = {f'{k}': v for k, v in sparsity_metrics.items()}
                log_stat.update(sparsity_log)
                net.dpd_model.backbone.set_debug(1)

    # Merge stat dicts into the log dict
    if train_stat is not None:
        train_stat_log = {f'TRAIN_{k.upper()}': v for k, v in train_stat.items()}
        log_stat = {**log_stat, **train_stat_log}
    if val_stat is not None:
        val_stat_log = {f'VAL_{k.upper()}': v for k, v in val_stat.items()}
        log_stat = {**log_stat, **val_stat_log}

    return log_stat


def gen_dir_paths(args: argparse.Namespace):
    # Include version in path if provided
    version_path = args.version if hasattr(args, 'version') and args.version else ''
    
    # if args.step == 'train_pa':
    save_base = os.path.join('./save')
    if version_path:
        save_base = os.path.join(save_base, version_path+'_'+args.dataset_name)
    else:
        save_base = os.path.join(save_base, args.dataset_name)
    # if args.quant_dir_label:
        # save_base = os.path.join(save_base, args.quant_dir_label)
    path_dir_save = save_base
    
    log_base = os.path.join('./log')
    if version_path:
        log_base = os.path.join(log_base, version_path+'_'+args.dataset_name)
    else:
        log_base = os.path.join(log_base, args.dataset_name)
    # if args.quant_dir_label:
        # log_base = os.path.join(log_base, args.quant_dir_label)
    path_dir_log = log_base

    terminal_log_base = os.path.join('./terminal_log')
    if version_path:
        terminal_log_base = os.path.join(terminal_log_base, version_path+'_'+args.dataset_name)
    else:
        terminal_log_base = os.path.join(terminal_log_base, args.dataset_name)
    # if args.quant_dir_label:
        # terminal_log_base = os.path.join(terminal_log_base, args.quant_dir_label)
    path_dir_terminal_log = terminal_log_base

    # elif args.step == 'train_dpd' or args.step == 'run_dpd':
    #     # Organize DPD files under PA model directory
    #     pa_model_id = gen_pa_model_id(args)
    #     base_path = os.path.join('./save', args.dataset_name, args.step, pa_model_id)
    #     if version_path:
    #         base_path = os.path.join(base_path, version_path)
    #     if args.quant_dir_label:
    #         base_path = os.path.join(base_path, args.quant_dir_label)
    #     path_dir_save = base_path
        
    #     log_base = os.path.join('./log', args.dataset_name, args.step, pa_model_id)
    #     if version_path:
    #         log_base = os.path.join(log_base, version_path)
    #     if args.quant_dir_label:
    #         log_base = os.path.join(log_base, args.quant_dir_label)
    #     path_dir_log_hist = os.path.join(log_base, 'history')
    #     path_dir_log_best = os.path.join(log_base, 'best')
    dir_paths = (path_dir_save, path_dir_log, path_dir_terminal_log)
    return dir_paths


def gen_file_paths(path_dir_save: str, path_dir_log: str, path_dir_terminal_log: str, model_id: str, step: str):
    # File Paths
    path_file_save_best = os.path.join(path_dir_save, model_id + '.pt')
    path_file_log_hist = os.path.join(path_dir_log, step, 'history.csv')
    path_file_log_best = os.path.join(path_dir_log, step, 'best.csv')
    path_file_terminal_log = os.path.join(path_dir_terminal_log, step + '.log')
    file_paths = (path_file_save_best, path_file_log_hist, path_file_log_best, path_file_terminal_log)
    return file_paths


def create_folder(folder_list):
    for folder in folder_list:
        try:
            os.makedirs(folder)
        except:
            pass

# def gen_pa_model_id(args):
#     dict_pa = {'S': f"{args.seed}",
#                'M': args.PA_backbone.upper(),
#                'H': f"{args.PA_hidden_size:d}",
#                'L': f"{args.PA_num_layers:d}",                  # n_net_pa_params 추가하기 (.pt 파일 저장 코드 참고)
#                }
#     dict_pamodel_id = dict(list(dict_pa.items()))

#     # PA Model ID
#     list_pamodel_id = []
#     for item in list(dict_pamodel_id.items()):
#         list_pamodel_id += list(item)
#     pa_model_id = '_'.join(list_pamodel_id)
#     pa_model_id = 'PA_' + pa_model_id
#     return pa_model_id