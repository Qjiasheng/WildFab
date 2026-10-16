import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import sdf_meshing
import configargparse
from train_udf import UDF

class UDFDecoder(torch.nn.Module):
    def __init__(self, model_type='sine', ckpt_path=None):
        super().__init__()
        self.model = UDF(model_type=model_type)
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))

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
    p.add_argument('--sample_num', type=int, default=150000, help='number of points to sample for tracing')
    opt = p.parse_args()

    # General training options in config file
    configs = utils.load_configs(opt.config_path, opt.experiment_name)
    dataset_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']

    udf_decoder = UDFDecoder(model_type=optim_paras.get('model_type'), ckpt_path=dataset_paras.get('udf_checkpoint_path'))

    save_path = os.path.join(dataset_paras.get('save_path'), opt.experiment_name)
    utils.cond_mkdir(save_path)

    # wn volume and wn gradnorm volume
    sdf_meshing.create_wn_volume(save_path=save_path, dataset_paras=dataset_paras, N=dataset_paras.get('resolution'))

    # udf volume and udf gradnorm volume
    sdf_meshing.create_udf_volume(udf_decoder, save_path=save_path, dataset_paras=dataset_paras, N=dataset_paras.get('resolution'))

    # trace surface points in min_udf, min_udf_gradnorm, and wn_gradnorm
    sdf_meshing.surf_extraction_volume_wn(udf_decoder, save_path=save_path, dataset_paras=dataset_paras, 
                                          spec_N=256, sample_num=opt.sample_num, enable_eval_metrics=True,
                                          wn_gradnorm_valid_th=10.0)  


