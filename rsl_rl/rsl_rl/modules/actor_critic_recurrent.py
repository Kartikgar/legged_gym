# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories
from .vision_encoder import Encoder, Mlp
class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
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

        super().__init__(num_actor_obs=rnn_hidden_size,
                         num_critic_obs=rnn_hidden_size,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std)

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

    def get_latent(self,observations):
        if len(observations.shape)<3:
            prop_a = self.mlp_a.forward(observations[:,:48])
            img_a = self.feature_extractor_a.forward(observations[:,48:].reshape((observations.size(0),1,32,32)))
            out = torch.hstack((prop_a,img_a))
        else:
            observations_re = observations.reshape((observations.shape[0]*observations.shape[1]),observations.shape[2])
            prop_a = self.mlp_a.forward(observations_re[:,:48])
            img_a = self.feature_extractor_a.forward(observations_re[:,48:].reshape((observations_re.size(0),1,32,32)))
            out = torch.hstack((prop_a,img_a))
            out =out.reshape((observations.shape[0],observations.shape[1],out.shape[-1]))
            # import pdb;pdb.set_trace()
        return out
    
    def act(self, observations, masks=None, hidden_states=None):
        # print("observations:", observations.shape)
        out = self.get_latent(observations)
        input_a = self.memory_a(out, masks, hidden_states)
        # print ("actor rnn output:", input_a.shape)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        # prop_a = self.mlp_a.forward(observations[:,:48])
        # img_a = self.feature_extractor_a.forward(observations[:,48:].reshape((observations.size(0),1,32,32)))
        # out = torch.hstack((prop_a,img_a))
        out = self.get_latent(observations)
        input_a = self.memory_a(out)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        # prop_c = self.mlp_a.forward(critic_observations[:,:48])
        # img_c = self.feature_extractor_a.forward(critic_observations[:,48:].reshape((critic_observations.size(0),1,32,32)))
        # out = torch.hstack((prop_c,img_c))
        out = self.get_latent(critic_observations)
        input_c = self.memory_c(out, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
    
    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0

if __name__ == '__main__':

    model = ActorCriticRecurrent(num_actor_obs = 48 , num_critic_obs=48,
                            num_actions = 10, 
                            actor_hidden_dims = [256,256,256], 
                            activation='elu')
    print (model)
    input = torch.randn(10,1072)
    out = model(input)
    import pdb;pdb.set_trace()