import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio
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

    # build waypt data and relate time-octree
    waypoint_path = dataset_paras.get('waypoint_path')
    waydata = dataio.WayPts(waypoint_path=waypoint_path)

    # set batchsize for time-varying query
    batch_size = 10
    sdf_meshing.create_tv_wn_volume(waydata=waydata, batch_size=batch_size, dataset_paras=dataset_paras,
                                    save_path=save_path, N=256,)


    # cmd
    # python toy_tests/fastwn_tv.py --experiment_name=klein_single_shell --config_path=configs/collision_config.yaml 