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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, Logger
from legged_gym.utils.task_registry_rma import task_registry
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
import open3d as o3d
def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def play(args):
    # if args.web:
    #     web_viewer = webviewer.WebViewer()
    # faulthandler.enable()
    # exptid = args.exptid
    # log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    all_actions =[]
    # override some parameters for testing
    # if args.nodelay:
    #     env_cfg.domain_rand.action_delay_view = 0
    # voxel_size = 1
    # nandi_path = "/home/kartik/visualising_pcl/pcl/09-09-14-23-15_filtered.pcd"
    # nandi_geom =o3d.io.read_point_cloud(nandi_path)
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(nandi_geom, voxel_size)
    # voxel_list = voxel_grid.get_voxels()
    # coords =[]
    # for i in range(len(voxel_list)):
    #     coords.append(voxel_list[i].grid_index.reshape((3,1)))
    # coords = np.array(coords)
    # coords = np.reshape(coords,(coords.shape[0],3))
    # terrain_width = coords[:,1].max() - coords[:,1].min()
    # terrain_length = coords[:,0].max() - coords[:,0].min()
    # horizontal_scale = voxel_size  # [m]
    # vertical_scale = voxel_size  # [m]
    # num_rows = int(terrain_width/horizontal_scale)
    # num_cols = int(terrain_length/horizontal_scale)
    # env_cfg.env.num_envs = 16 if not args.save else 64
    # env_cfg.env.episode_length_s = 60
    # env_cfg.commands.resampling_time = 60
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 5
    # # env_cfg.terrain.horizontal_scale = voxel_size
    # # env_cfg.terrain.vertical_scale = voxel_size
    # env_cfg.terrain.height = [0.02, 0.02]
    # env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
    #                                 "rough slope up": 0.0,
    #                                 "rough slope down": 0.0,
    #                                 "rough stairs up": 0., 
    #                                 "rough stairs down": 0., 
    #                                 "discrete": 0., 
    #                                 "stepping stones": 0.0,
    #                                 "gaps": 0., 
    #                                 "smooth flat": 0,
    #                                 "pit": 0.0,
    #                                 "wall": 0.0,
    #                                 "platform": 0.,
    #                                 "large stairs up": 0.,
    #                                 "large stairs down": 0.,
    #                                 "parkour": 0.0,
    #                                 "parkour_hurdle": 0.0,
    #                                 "parkour_flat": 0.,
    #                                 "parkour_step": 0.0,
    #                                 "parkour_gap": 0.0, 
    #                                 "demo": 1,
    #                                 "hill": 0}
    
    # env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    # env_cfg.terrain.curriculum = False
    # env_cfg.terrain.max_difficulty = True
    
    # env_cfg.depth.angle = [0, 1]
    # env_cfg.noise.add_noise = True
    # env_cfg.domain_rand.randomize_friction = True
    # env_cfg.domain_rand.push_robots = False
    # env_cfg.domain_rand.push_interval_s = 6
    # env_cfg.domain_rand.randomize_base_mass = False
    # env_cfg.domain_rand.randomize_base_com = False

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # import pdb;pdb.set_trace()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    if EXPORT_POLICY:
        path = os.path.join(log_pth, "traced")
        # import pdb;pdb.set_trace()
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.perception.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None
    print(policy)
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    for i in range(10*int(env.max_episode_length)):
        if EXPORT_POLICY:
            if env.cfg.perception.use_camera:
                if infos["depth"] is not None:
                    depth_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
                    actions, depth_latent = policy_jit(obs.detach(), True, infos["depth"], depth_latent)
                else:
                    depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
                    actions, depth_latent = policy_jit(obs.detach(), False, depth_buffer, depth_latent)
            else:
                obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
                actions = policy(obs_jit)
        else:
            if env.cfg.perception.use_camera:
                if infos["depth"] is not None:
                    obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5*yaw
                    
            else:
                depth_latent = None
            
            if hasattr(ppo_runner.alg, "depth_actor"):
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            else:
                # import pdb;pdb.set_trace()
                actions = policy(obs.detach(), hist_encoding=False, scandots_latent=depth_latent)
            
        obs, _, rews, dones, infos = env.step(actions.detach())
        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        # if args.web:
        #     web_viewer.render(fetch_results=True,
        #                 step_graphics=True,
        #                 render_all_camera_sensors=True,
        #                 wait_for_page_load=True)
        # print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
        #       "cmd vx", env.commands[env.lookat_id, 0].item(),
        #       "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
        # print(depth_encoder)
        # id = env.lookat_id
        

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
