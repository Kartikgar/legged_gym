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

class Stoch3RoughCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 1024
        num_actions = 12

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = True

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        pi = 3.14
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'bl_abd_joint': 0.0,   # [rad]
            'bl_hip_joint': pi/4,   # [rad]
            'bl_knee_joint': -pi/2 ,  # [rad]

            'br_abd_joint': 0.0,   # [rad]
            'br_hip_joint': pi/4,     # [rad]
            'br_knee_joint': -pi/2,   # [rad]

            'fl_abd_joint': 0.0,     # [rad]
            'fl_hip_joint': pi/4,   # [rad]
            'fl_knee_joint': -pi/2,   # [rad]
            
            'fr_abd_joint': 0.0,    # [rad]
            'fr_hip_joint': pi/4,  # [rad]
            'fr_knee_joint': -pi/2,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 80.}  # [N*m/rad]
        damping = {'joint': 2.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.15
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/stoch3/urdf/stoch3.urdf'
        name = "stoch3"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "shank"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.45
        max_contact_force = 500.
        class scales( LeggedRobotCfg.rewards.scales ):
            pass

class Stoch3RoughCfgPPO( LeggedRobotCfgPPO ):

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_stoch3'
        load_run =-1
        
  