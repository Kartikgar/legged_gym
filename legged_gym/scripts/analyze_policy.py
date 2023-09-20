import torch
from rsl_rl.modules import vision_encoder
from rsl_rl.modules import ActorCritic
import torch
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.nn import Sequential
from torch.nn.functional import normalize
class analyzed(Module):
    def __init__(self, channels =1, outDims=64):

        super(analyzed, self).__init__()
        img_out_dims = 64
        self.feature_extraction_f = vision_encoder.Encoder(channels=1,outDims=img_out_dims)
        self.feature_extraction_b = vision_encoder.Encoder(channels=1,outDims=img_out_dims)
    def forward(self,x):

        img_f = self.feature_extraction_f.forward(x[:,48:48+32*32].reshape((x.size(0),1,32,32)))
        img_b = self.feature_extraction_b.forward(x[:,48+32*32:].reshape((x.size(0),1,32,32)))
        out = torch.hstack((img_f,img_b))

        # print (" i am being used :)")
        return out

def rename_dict(state_dict):
    new_k = {'actor.'+ j for j in model.state_dict().keys() }




if __name__=="__main__":
    model = analyzed()
    num_critic_obs = 48+32*32 + 32*32
    num_actor_obs = 48
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'
    model_ac = ActorCritic(num_actor_obs,
                        num_critic_obs,
                        12,
                        actor_hidden_dims,
                        critic_hidden_dims,
                        activation='elu',
                        init_noise_std=1.0,)
    model_dict = model.state_dict()
    weights_path = "/home/kartik/legged_gym/logs/rough_go1/Sep14_19-37-05_go1_cnn_32x32_fb/model_1500.pt"
    state_dict = torch.load(weights_path,map_location=torch.device(0))
    model_ac.load_state_dict(state_dict['model_state_dict'])
    pre_dict = { k: v for k, v in state_dict['model_state_dict'].items() if k in model_ac.actor.sate_dict().keys()}
    # final_dict = dict(zip(old_k, list(pre_dict.values())))
    model.load_state_dict(pre_dict)
    import pdb;pdb.set_trace()