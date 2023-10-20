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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class A1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
    class env:
        num_envs = 4096
        n_scan = 187
        n_priv = 3+3+3
        n_priv_latent = 4+1
        n_proprio = 48
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        history_encoding = True

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = True
        curriculum=True
        # terrain type :[smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap, pit]
        terrain_dict = {"rough_slopes":0.2,
                        "slope":0,
                        "stairs_up":0.2,
                        "stairs_down":0.2,
                        "discrete":0.2,
                        "stepping_stones":0.0,
                        "gap":0.0,
                        }
        # terrain_proportions = [0.15, 0, 0.15, 0.10, 0.20,.10,0.15] 
        terrain_proportions = list(terrain_dict.values())
        height = [0.02, 0.04]
        # terrain_proportions=[0]*7
        # terrain_length = 18.
        # terrain_width = 4
        # num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        # num_cols = 40 # number of terrain cols (types)
    class commands:
        curriculum = False
        reindex = False
        max_curriculum = 1.
        
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.4, 0.5] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-.5, .5]    # min max [rad/s]
            heading = [-3.14/6, 3.14/6]
            # heading =[0,0]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 2.25]
        randomize_base_mass = True
        added_mass_range = [0.,3.0]
        randomize_base_com = True
        added_com_range = [-0.2, 0.2]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_motor = False
        motor_strength_range = [0.8, 1.2]

        delay_update_global_steps = 24 * 8000
        action_delay = False
        action_curr_step = [1, 1]
        action_curr_step_scratch = [0, 1]
        action_delay_view = 1
        action_buf_len = 8  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.33
        max_contact_force = 50
        only_positive_rewards = True
        class scales( LeggedRobotCfg.rewards.scales ):

            torques = -0.000001
            dof_pos_limits = -10.0
            feet_step = 0.0
            stumble = -0.0
            collision = -2.5
            stand_still = -0.0
            termination = 0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # lin_vel_z = -0.0
            # ang_vel_xy = -0.00
            orientation = -0.0
            # dof_vel = -0.0007
            dof_acc = -2.5e-7
            base_height = -5. 
            feet_air_time =  1.0
            action_rate = -0.01
            work = -0.00000
            forward_vel = 0
            angular_vel =0.
            feet_drag_penalty = -0.00000
            command_tracking = 0
            feet_contact_forces = -0.00
            alive = 0.0
    

    class perception(LeggedRobotCfg.perception):
        compute_depth=False
        compute_segmentation=False
        use_camera = False
        # camera_names=['front', 'bottom']
        # camera_names=['front']
        # camera_poses = [[0.3, 0, 0], [0, 0.15, 0], [0, -0.15, 0], [0.1, 0, -0.1], [-0.2, 0, -0.1]]
        # camera_rpys = [[0.0, 0 , 0], [0, 0, 3.14 / 2], [0, 0, -3.14 / 2], [0, -3.14 / 2, 0],
        #                [0, -3.14 / 2, 0]]
        # camera_poses= [[0.3, 0, 0],[0,1.0,-0.1]]
        # camera_rpys = [[0.0, 0 , 0],[0,-3.14/2,0]]
        camera_poses= [[0.3, 0, 0]]
        camera_rpys = [[0.0, 0 , 0]]
        resized = (32, 32)
        # image_height=32
        # image_width=32

class A1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_a1'

        

  