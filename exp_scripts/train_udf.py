import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
import shutil
import torch

# set seed
utils.set_seed(124)

class UDF(torch.nn.Module):
    def __init__(self, model_type='sine'):
        super().__init__()
        # only outmost_linear False and outer_nl given -- valid set outmost nl activation
        self.model = modules.SingleBVPNet(type=model_type, in_features=3, out_features=1, 
                                          hidden_features=256, num_hidden_layers=3, 
                                          outermost_linear=False, outer_nl='softplus')  
        self.model.cuda()

    def forward(self, model_in):
        return self.model(model_in)

def no_outer_batch(batch):
    """remove outer batch, since each batch is already a batch of waypts"""
    return batch[0]


if __name__ == '__main__':

    p = configargparse.ArgumentParser()

    p.add_argument('--experiment_name', type=str, required=True,
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    p.add_argument('--config_path', default=None, help='config file for training')
    opt = p.parse_args()

    # General training options in config file
    configs = utils.load_configs(opt.config_path, opt.experiment_name)
    dataset_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']

    udf_dataset = dataio.UDFDatasetDiffuse(dataset_paras.get('point_cloud_path'), 
                                           more_smpl_times=dataset_paras.get('more_smpl_times', 0),
                                           near_surf_times=dataset_paras.get('near_surf_times', 4), 
                                           near_inv_delta=dataset_paras.get('near_inv_delta', 200),
                                           far_surf_times=dataset_paras.get('far_surf_times', 6),
                                           far_inv_delta=dataset_paras.get('far_inv_delta', 128),
                                           use_adaptive_near_far=dataset_paras.get('use_adaptive_near_far', False),

                                           batch_size=dataset_paras.get('batch_size'),
                                           down_ratio=dataset_paras.get('downsample'),
                                           high_feature_ratio=dataset_paras.get('high_feature_ratio'),
                                           sample_type=dataset_paras.get('sample_type', 'non-random'),)
    
    dataloader = DataLoader(udf_dataset, shuffle=True, batch_size=1, pin_memory=False, 
                            num_workers=0, collate_fn=no_outer_batch)
    # next(iter(dataloader))  # test dataloader

    # Define the model.
    model = UDF(model_type=optim_paras.get('model_type'))
    model.train()

    # Define the loss
    loss_fn = loss_functions.udf_scalar_diffuse_dirichlet
    summary_fn = utils.write_sdf_summary

    save_path = os.path.join(dataset_paras.get('save_path'), opt.experiment_name)
    if os.path.exists(save_path):
        val = input("The model directory %s exists. Overwrite? (y/n)"%save_path)
        if val == 'y':
            shutil.rmtree(save_path)

    os.makedirs(save_path)
    utils.put_configs(os.path.join(dataset_paras.get('save_path'), opt.experiment_name, 'train_configs.yaml'), configs) 

    training.train(model=model, train_dataloader=dataloader, epochs=optim_paras.get('num_epochs'), lr=optim_paras.get('lr'),
                steps_til_summary=optim_paras.get('steps_til_summary'), epochs_til_checkpoint=optim_paras.get('epochs_til_ckpt'),
                model_dir=save_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
                clip_grad=True, loss_paras=loss_paras)



