import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import configargparse
import level_collision
from test_udf import UDFDecoder
from train_collision import Quat

class QuatDecoder(torch.nn.Module):
    def __init__(self, model_type='sine', checkpont_path=None):
        super().__init__()
        self.model = Quat(model_type=model_type)
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
    p.add_argument('--config_path', default=None, help='config file for training.')
    opt = p.parse_args()

    # General training options in config file
    configs = utils.load_configs(opt.config_path, opt.experiment_name)
    data_paras, optim_paras, loss_paras = configs['dataset'], configs['optim'], configs['parameters']


    # prepare decoders
    udf_decoder = UDFDecoder(ckpt_path=data_paras.get('udf_checkpoint_path'))

    quat_path = data_paras.get('collision_checkpoint_path')
    if quat_path is None or not os.path.exists(quat_path):
        # if not given, or test w/o training, use print_dirs and one refer dir for local frame
        quat_decoder = None   
    else:
        quat_decoder = QuatDecoder(checkpont_path=data_paras.get('collision_checkpoint_path'))

    save_path = os.path.join(data_paras.get('save_path'), opt.experiment_name)


    # waypts collision test after optimization
    # batch_size is chunk of waypoints, and max_batch is chunk of query points 
    # see other specific para in function

    # if checking collision w/o any optimization, set quat_decoder=None
    level_collision.quat_collision_test(udf_decoder, quat_decoder, data_paras, loss_paras, save_path,
                                        batch_size=15000, max_batch=96**3)

