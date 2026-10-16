# script to use pre-trained SDF and heightfield as initialisation for a scalar field training
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import configargparse
import dataio, training
from torch.utils.data import DataLoader
import loss_functions
import shutil
from test_udf import UDFDecoder
from test_slice import SliceDecoder
from train_init_slice import Slice

# set seed
utils.set_seed(124)

    
if __name__ == '__main__':

    p = configargparse.ArgumentParser()

    p.add_argument('--experiment_name', type=str, required=True,
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    p.add_argument('--config_path', default=None, help='config file for training')
    opt = p.parse_args()

    # General training options in config file
    configs = utils.load_configs(opt.config_path, opt.experiment_name)
    dataset_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']

    udf_decoder = UDFDecoder(model_type=optim_paras.get('model_type'), ckpt_path=dataset_paras.get('udf_checkpoint_path'))
    preslice_decoder = SliceDecoder(model_type=optim_paras.get('model_type'), checkpont_path=dataset_paras.get('init_checkpoint_path'))

    slice_dataset = dataio.FinalSlicePCD(udf_decoder=udf_decoder,
                                        preslice_decoder=preslice_decoder,
                                        pointcloud_path=dataset_paras.get('point_cloud_path'), 
                                        infill_surf_path=dataset_paras.get('infill_surf_path'),  # if none, no infills
                                        sample_number=dataset_paras.get('batch_size'), 
                                        num_batches=dataset_paras.get('num_batches'),
                                        infill_times=dataset_paras.get('infill_times', 1.0),
                                        down_ratio=dataset_paras.get('downsample'),
                                        base_threshold=dataset_paras.get('base_threshold'),
                                        large_th_pcd_index_path=dataset_paras.get('large_th_pcd_index_path', None),
                                        stage='final')

    dataloader = DataLoader(slice_dataset, shuffle=True, batch_size=1, pin_memory=False, num_workers=0)


    next(iter(dataloader))  # test dataloader

    # 2-manifold guidance field NN using height field initialisation
    slice_model = Slice(model_type=optim_paras.get('model_type'))
    slice_model.load_state_dict(torch.load(dataset_paras.get('init_checkpoint_path')))  
    slice_model.cuda()
    slice_model.train()


    # define loss function
    loss_fn = loss_functions.final_slice_field  
    summary_fn = utils.write_sdf_summary


    save_path = os.path.join(dataset_paras.get('save_path'), opt.experiment_name)
    if os.path.exists(save_path):
        val = input("The model directory %s exists. Overwrite? (y/n)"%save_path)
        if val == 'y':
            shutil.rmtree(save_path)

    os.makedirs(save_path)
    utils.put_configs(os.path.join(dataset_paras.get('save_path'), opt.experiment_name, 'train_configs.yaml'), configs) 

    training.train(model=slice_model, train_dataloader=dataloader, epochs=optim_paras.get('num_epochs'), lr=optim_paras.get('lr'),
                steps_til_summary=optim_paras.get('steps_til_summary'), epochs_til_checkpoint=optim_paras.get('epochs_til_ckpt'),
                model_dir=save_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
                clip_grad=True, loss_paras=loss_paras)



