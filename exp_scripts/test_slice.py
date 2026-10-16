import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import configargparse
import slice_meshing
from train_init_slice import Slice

class SliceDecoder(torch.nn.Module):
    def __init__(self, model_type='sine', checkpont_path=None):
        super().__init__()
        self.model = Slice(model_type=model_type)
        if checkpont_path is not None:
            self.model.load_state_dict(torch.load(checkpont_path))

        self.model.cuda()
        self.model.eval()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)

if __name__ == '__main__':
    p = configargparse.ArgumentParser()

    p.add_argument('--experiment_name', type=str, required=True,
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    p.add_argument('--config_path', default=None, help='config file for training')
    opt = p.parse_args()

    configs = utils.load_configs(opt.config_path, opt.experiment_name)
    dataset_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']


    # slice field decoder
    slice_decoder = SliceDecoder(checkpont_path=dataset_paras.get('slice_checkpoint_path'))

    save_path = os.path.join(dataset_paras.get('save_path'), opt.experiment_name)

    # slice field volume query 
    slice_meshing.query_volume_field(slice_decoder, save_path=save_path, N=dataset_paras.get('resolution'))

    # steamlines on surface only when having mesh
    slice_meshing.streamline_on_surface(slice_decoder, dataset_paras, num_seeds=5000, show_mesh=True)

    # pcd query in slice field
    slice_meshing.query_pcd_field(slice_decoder, point_cloud_path=dataset_paras.get('point_cloud_path'))

    # show interior iso-layers clamped infill surface wn field
    slice_meshing.show_inside_isolayers(slice_decoder, infill_surf_path=dataset_paras.get('mesh_path'),  
                                        num_layers=60, N=dataset_paras.get('resolution'))
    

    # ----------------------------- sep function to append large layer thickness pcd for next fn training
    # to improve layer thinkness uniformity near field singulairties.
    append_large_th_pcd = dataset_paras.get('append_large_th_pcd', False)   # typically in fn config
    if append_large_th_pcd:
        slice_meshing.append_large_th_pcd(slice_decoder, point_cloud_path=dataset_paras.get('point_cloud_path'), 
                                          save_path=save_path, dataset_paras=dataset_paras, large_percent=0.02)
    


