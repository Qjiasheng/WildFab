import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import configargparse
import shutil
import slice_toolpath

from test_udf import UDFDecoder
from test_slice import SliceDecoder

utils.set_seed(0)
p = configargparse.ArgumentParser()


if __name__ == '__main__':

    p.add_argument('--experiment_name', type=str, required=True,
                    help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    p.add_argument('--config_path', default=None, help='config file for training')
    opt = p.parse_args()


    # General options in config file
    configs = utils.load_configs(opt.config_path, opt.experiment_name)
    data_paras, optim_paras, toolpath_paras, toolpath_type = configs['dataset'], configs['optim'], configs['parameters'], configs['type']

    udf_decoder = UDFDecoder(ckpt_path=data_paras.get('udf_checkpoint_path'))
    slice_decoder = SliceDecoder(checkpont_path=data_paras.get('slice_checkpoint_path'))

    save_path = os.path.join(data_paras.get('save_path'), opt.experiment_name)


    if os.path.exists(save_path):
        val = input("The model directory %s exists. Overwrite? (y/n)"%save_path)
        if val == 'y':
            shutil.rmtree(save_path)
        else:
            sys.exit()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    utils.put_configs(os.path.join(data_paras.get('save_path'), opt.experiment_name, 'toolpath_configs.yaml'), configs) # save to prj folder


    # toolpath generation 
    # single shell toolpath from either triangle soup iso-contouring or alternating UDF/GWN and guidance field projection
    # infills generation, contours planning can refer to INF-3DP, not provided here.

    # single shell surface toolpath generation with iso-contours from triangle soups
    slice_toolpath.single_shell_toolpath_gen_mesh(udf_decoder, slice_decoder, data_paras, toolpath_paras, save_path)

    # single shell surface toolpath generation with iso-contours from alternating UDF/GWN and guidance field projection
    # this algorithm in Paper [Algorithm 1]
    slice_toolpath.single_shell_toolpath_gen_field(udf_decoder, slice_decoder, data_paras, toolpath_paras, save_path)

