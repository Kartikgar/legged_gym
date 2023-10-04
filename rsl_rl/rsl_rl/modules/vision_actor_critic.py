import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .actor_critic import ActorCritic, get_activation
from .actor_critic_recurrent import ActorCriticRecurrent, Memory
from rsl_rl.utils import unpad_trajectories
from .vision_encoder import Encoder, Mlp
class VisualActorCriticRecurrent(ActorCriticRecurrent):
    is_recurrent = True
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        vis_laten
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)
        self.num_prop_obs =48
        self.num_img_obs = num_actor_obs - self.num_prop_obs
        super().__init__(num_prop_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims,
                        critic_hidden_dims,
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        **kwargs)

        activation = get_activation(activation)
        first_out_size =128
        self.mlp_a = Mlp(inDims =48,outDims=first_out_size)
        self.feature_extractor_a = Encoder(channels=1, outDims=first_out_size)
        # self.mlp_c = Mlp(48,first_out_size)
        # self.feature_extractor_c = Encoder(channels=1, outDims=first_out_size)
        self.memory_a = Memory(first_out_size*2, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(first_out_size*2, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        # self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        # self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        # print(f"Actor RNN: {self.memory_a}")
        # print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        # print("observations:", observations.shape)
        # if len(observations.shape)<3:
        #     prop_a = self.mlp_a.forward(observations[:,:48])
        #     img_a = self.feature_extractor_a.forward(observations[:,48:].reshape((observations.size(0),1,32,32)))
        # else:

        # out = torch.hstack((prop_a,img_a))
        input_a = self.memory_a(observations, masks, hidden_states)
        # print ("actor rnn output:", input_a.shape)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        # prop_a = self.mlp_a.forward(observations[:,:48])
        # img_a = self.feature_extractor_a.forward(observations[:,48:].reshape((observations.size(0),1,32,32)))
        # out = torch.hstack((prop_a,img_a))
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        # prop_c = self.mlp_a.forward(critic_observations[:,:48])
        # img_c = self.feature_extractor_a.forward(critic_observations[:,48:].reshape((critic_observations.size(0),1,32,32)))
        # out = torch.hstack((prop_c,img_c))
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))
    
if __name__ == '__main__':

    model = ActorCriticRecurrent(num_actor_obs = 48 , num_critic_obs=48,
                            num_actions = 10, 
                            actor_hidden_dims = [256,256,256], 
                            activation='elu')
    print (model)
    input = torch.randn(10,1072)
    out = model(input)
    import pdb;pdb.set_trace()