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

# set seed
utils.set_seed(1024)

class Slice(torch.nn.Module):
    def __init__(self, model_type='sine'):
        super().__init__()
        # only outmost_linear False and outer_nl given -- valid set outmost nl activation
        self.model = modules.SingleBVPNet(type=model_type, in_features=3, out_features=1, 
                                          hidden_features=256, num_hidden_layers=3, 
                                          outermost_linear=True)  
        self.model.cuda()

    def forward(self, model_in):
        return self.model(model_in)
    

    
if __name__ == '__main__':

    p = configargparse.ArgumentParser()
    p.add_argument('--experiment_name', type=str, required=True,
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    p.add_argument('--config_path', default=None, help='config file for training')
    opt = p.parse_args()

    # General training options in config file
    configs = utils.load_configs(opt.config_path, opt.experiment_name)
    dataset_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']

    slice_dataset = dataio.SliceSurfPCDHeat(pointcloud_path=dataset_paras.get('point_cloud_path'), 
                                            sample_number=dataset_paras.get('batch_size'), 
                                            num_batches=dataset_paras.get('num_batches'),
                                            down_ratio=dataset_paras.get('downsample'),
                                            base_threshold=dataset_paras.get('base_threshold'),
                                            stage='init')

    # adapted from INF-3DP, without collate_fn=no_outer_batch. 
    dataloader = DataLoader(slice_dataset, shuffle=True, batch_size=1, pin_memory=False, num_workers=0)

    # 2-manifold guidance field NN using height field initialisation
    # ! need height field as initialization, pls run train_height.py first
    slice_model = Slice(model_type=optim_paras.get('model_type'))
    slice_model.load_state_dict(torch.load(dataset_paras.get('height_checkpoint_path')))  
    slice_model.cuda()
    slice_model.train()


    # define loss function
    loss_fn = loss_functions.init_slice_field
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



