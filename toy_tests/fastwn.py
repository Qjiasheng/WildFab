import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import sdf_meshing
import configargparse


if __name__ == '__main__':

    p = configargparse.ArgumentParser()
    p.add_argument('--experiment_name', type=str, required=True,
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    p.add_argument('--config_path', default=None, help='config file for training')
    opt = p.parse_args()

    # General training options in config file
    configs = utils.load_configs(opt.config_path, opt.experiment_name)
    dataset_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']

    save_path = os.path.join(dataset_paras.get('save_path'), opt.experiment_name)
    utils.cond_mkdir(save_path)

    # wn volume and wn gradnorm volume
    sdf_meshing.create_wn_volume(save_path=save_path, dataset_paras=dataset_paras, N=dataset_paras.get('resolution'))



    # cmd 
    # python toy_tests/fastwn.py --experiment_name=fertility_pcd --config_path=configs/udf_config.yaml 