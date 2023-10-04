from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.terrain_utils import *
from rsl_rl.modules import vision_encoder
from rsl_rl.modules import ActorCritic
from torch.nn import Module

from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.nn import Sequential
from torch.nn.functional import normalize
import numpy as np
import imageio
import math
import os

import matplotlib.pyplot as plt
import torch

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

    model = torch.load('/home/kartik/legged_gym/legged_gym/scripts/rnn_gru_1.pth')
    model.eval()
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()

    #setting common params
    sim_params.dt = 1 / 60
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0)

    # set PhysX-specific parameters
    sim_params.physx.use_gpu =True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    # set Flex-specific parameters
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 20
    sim_params.flex.relaxation = 0.8
    sim_params.flex.warm_start = 0.5

    sim_params.use_gpu_pipeline = True

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    # configure the ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0

    # create the ground plane
    num_terains = 9
    num_sqrt = int(np.sqrt(num_terains))
    terrain_width = 12.
    terrain_length = 12.
    horizontal_scale = 0.25  # [m]
    vertical_scale = 0.005  # [m]
    num_rows = int(terrain_width/horizontal_scale)
    num_cols = int(terrain_length/horizontal_scale)
    heightfield = np.zeros((num_sqrt*num_rows, num_sqrt*num_cols), dtype=np.int16)

    def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
    for i in range(num_sqrt):
        for j in range(num_sqrt):
                if i%3 == 0:
                    heightfield[i*num_rows:(i+1)*num_rows, j*num_cols:(j+1)*num_cols] = random_uniform_terrain(new_sub_terrain(), min_height=-0.0, max_height=0.0, step=0.2, downsampled_scale=0.5).height_field_raw
                if i%3 ==1:
                    heightfield[i*num_rows:(i+1)*num_rows, j*num_cols:(j+1)*num_cols] = stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=-0.25).height_field_raw 
                if i%3 ==2:
                    heightfield[i*num_rows:(i+1)*num_rows, j*num_cols:(j+1)*num_cols] = stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=+0.25).height_field_raw -750
    vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = vertices.shape[0]
    tm_params.nb_triangles = triangles.shape[0]
    tm_params.transform.p.x = -1.
    tm_params.transform.p.y = -1.
    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)
    # gym.add_ground(sim, plane_params)

    # loading assets (e.g. robots)
    asset_root = '/home/kartik/Downloads/IsaacGym_Preview_4_Package/isaacgym/assets/urdf/'
    asset_file ='go1/urdf/go1.urdf'
    asset_options = gymapi.AssetOptions()
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    #creating env
    # set up the env grid
    num_envs = 1
    envs_per_row = 8
    env_spacing = 2.0
    env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

    # cache some common handles for later use
    envs = []
    actor_handles = []

    # create and populate the environments
    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
        envs.append(env)

        # height = random.uniform(1.0, 2.5)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.4)

        actor_handle = gym.create_actor(env, asset, pose, "MyActor", i, 1)
        actor_handles.append(actor_handle)
    
    camera_props = gymapi.CameraProperties()
    camera_props.enable_tensors = True
    camera_props.width = 32
    camera_props.height = 32
    # camera_props.use_collision_geometry = False

    cam_handle = gym.create_camera_sensor(env, camera_props)
    

    local_transform = gymapi.Transform()
    local_transform.p = gymapi.Vec3(0.3,0.,0.)
    local_transform.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
    gym.attach_camera_to_body(cam_handle, env, 0, local_transform, gymapi.FOLLOW_TRANSFORM)
    # cam_handle = gym.create_camera_sensor(env, camera_props)
    camera_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
    torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)
    gym.prepare_sim(sim)
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "forward")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "left")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "right")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "back")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "up")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "down")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "roll")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "pitch")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Y, "yaw")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "reset")
    
    actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(actor_root_state)
    
    initial_states = torch.Tensor([1,1,1.4,0,0,0,0,0,0,0,0,0,0]).to(device='cuda:0')
    # _rb_states = gym.get_actor_rigid_body_states(sim)
    # rb_states = gymtorch.wrap_tensor(_rb_states)
    while not gym.query_viewer_has_closed(viewer):
        gym.refresh_actor_root_state_tensor(sim)
        # print(root_states[0][0])
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action=="reset" and evt.value>0:
                gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(initial_states))
            elif evt.action=="forward" and evt.value>0:
                root_states[0][0] +=1
                gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
            elif evt.action=="back" and evt.value>0:
                root_states[0][0] -=1
                gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
            elif evt.action=="left" and evt.value>0:
                root_states[0][1] +=1
                gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
            elif evt.action=="right" and evt.value>0:
                root_states[0][1] -=1
                gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
            elif evt.action=="up" and evt.value>0:
                root_states[0][2] +=1
                gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
            elif evt.action=="down" and evt.value>0:
                root_states[0][2] -=1
                gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
            elif evt.action=="roll" and evt.value>0:
                root_states[0][1] +=1
                gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
                # import pdb;pdb.set_trace()
            # elif evt.action == "forward" and evt.value>0:




        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.render_all_camera_sensors(sim)
        gym.start_access_image_tensors(sim)
        camera_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
        torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
        torch_camera_tensor = torch.clamp(torch_camera_tensor, min =-10.0, max =0.0)
        torch_camera_tensor = torch_camera_tensor*0.1
        # print(torch_camera_tensor)
        ix=1
        with torch.no_grad():

            img_cnn = model.feature_extractor_a.feature_extraction(torch_camera_tensor.unsqueeze(0).unsqueeze(0))
        for j in range(img_cnn.shape[1]//5):
           for k in range(img_cnn.shape[1]//6):
               ax = plt.subplot(6,5,ix)
               plt.imshow(img_cnn[0,ix-1,:,:].cpu())
               ix+=1
        plt.pause(0.0000001)
        # import pdb;pdb.set_trace()
        gym.end_access_image_tensors(sim)


        # Wait for dt to elapse in real time.
        # This synchronizes the phy
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    # model = analyzed()
    # num_critic_obs = 48+32*32 + 32*32
    # num_actor_obs = 48
    # init_noise_std = 1.0
    # actor_hidden_dims = [512, 256, 128]
    # critic_hidden_dims = [512, 256, 128]
    # activation = 'elu'
    # model_ac = ActorCritic(num_actor_obs,
    #                     num_critic_obs,
    #                     12,
    #                     actor_hidden_dims,
    #                     critic_hidden_dims,
    #                     activation='elu',
    #                     init_noise_std=1.0,)
    # model_dict = model.state_dict()
    model = torch.load('/home/kartik/legged_gym/legged_gym/scripts/rnn_gru_1.pth')
    # weights_path = "/home/kartik/legged_gym/logs/rough_go1/Sep21_19-12-24_go1_gru_img/model_1500.pt"
    # state_dict = torch.load(weights_path,map_location=torch.device(0))
    import pdb;pdb.set_trace()
    # model_ac.load_state_dict(state_dict['model_state_dict'])
    # pre_dict = { k: v for k, v in state_dict['model_state_dict'].items() if k in model_ac.actor.sate_dict().keys()}
    # final_dict = dict(zip(old_k, list(pre_dict.values())))
    # model.load_state_dict(pre_dict)
