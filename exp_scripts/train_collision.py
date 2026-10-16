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

class Quat(torch.nn.Module):
    def __init__(self, model_type='sine'):
        super().__init__()
        self.model = modules.SingleBVPNet(type=model_type, in_features=3, out_features=4, 
                                          hidden_features=256, num_hidden_layers=3, 
                                          outermost_linear=True)  
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
    p.add_argument('--config_path', default=None, help='config file for training.')
    opt = p.parse_args()

    # General training options in config file
    configs = utils.load_configs(opt.config_path, opt.experiment_name)
    data_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']

    # prepare decoders
    udf_decoder = UDFDecoder(ckpt_path=data_paras.get('udf_checkpoint_path'))
    # load nozzle data 
    # less edge will not reduce pcd grads, but reduce gwn querying number and memory
    nozzle_coords, nozzle_grads, local_map = \
        utils.load_nozzle_data(data_paras.get('nozzle_nodes_path'), data_paras.get('nozzle_elems_path'),
        object_scale=loss_paras.get('object_scale'), sample_num=5000, device='cuda')

    # dataset
    collision_dataset = dataio.CollisionDataset(waypoint_path=data_paras.get('waypoint_path'),
                                                object_scale=loss_paras.get('object_scale'),
                                                batch_size=data_paras.get('batch_size'),
                                                shell_infill_ratio=data_paras.get('shell_infill_ratio', 4.0))
    
    dataloader = DataLoader(collision_dataset, shuffle=True, batch_size=1, pin_memory=False, 
                            num_workers=0, collate_fn=no_outer_batch)
    
    # next(iter(dataloader))  # test dataloader


    # Quaternion field for collision, out_features = 4 
    quater_model = Quat(model_type=optim_paras.get('model_type'))
    quater_model.train()

    # Define the loss
    loss_fn = loss_functions.collision_loss
    summary_fn = utils.write_collision_summary

    save_path = os.path.join(data_paras.get('save_path'), opt.experiment_name)
    if os.path.exists(save_path):
        val = input("The model directory %s exists. Overwrite? (y/n)"%save_path)
        if val == 'y':
            shutil.rmtree(save_path)

    os.makedirs(save_path)
    utils.put_configs(os.path.join(data_paras.get('save_path'), opt.experiment_name, 'trainconfigs.yaml'), configs) 

    # other needed paras
    # near surface within 5 mm as initial filter . if nozzle samples are sparse, set larger 
    # check the dist level set in viewer 
    udf_margin = torch.tensor(5.0 / loss_paras.get('object_scale')).float().cuda()
    loss_paras.update({'udf_init_margin': udf_margin})
    loss_paras.update({'udf_decoder': udf_decoder})
    loss_paras.update({'base_threshold': data_paras.get('base_threshold')}) # base waypts no sf loss, and upz loss
    
    # nozzle dict
    nozzle_dict = {'nozzle_coords': nozzle_coords, 'nozzle_grads': nozzle_grads, 'local_map': local_map}
    loss_paras.update({'nozzle_dict': nozzle_dict})
    
    training.train(model=quater_model, train_dataloader=dataloader, epochs=optim_paras.get('num_epochs'), lr=optim_paras.get('lr'),
                steps_til_summary=optim_paras.get('steps_til_summary'), epochs_til_checkpoint=optim_paras.get('epochs_til_ckpt'),
                model_dir=save_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
                clip_grad=True, loss_paras=loss_paras, collision_training=True)


