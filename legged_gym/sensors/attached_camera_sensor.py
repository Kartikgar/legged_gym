from .sensor import Sensor
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, filters, segmentation
from skimage.util import img_as_float32, img_as_float64
from skimage.segmentation import slic as sk_slic
from skimage.segmentation import mark_boundaries
from cuda_slic.slic import slic as cuda_slic
from fast_slic.avx2 import SlicAvx2
from torch.nn.functional import normalize

class AttachedCameraSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

        self.imgs=[]

    def initialize(self, camera_label, camera_pose, camera_rpy, env_ids=None):
        if env_ids is None: env_ids = range(self.env.num_envs)

        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.use_collision_geometry = True
        camera_props.width = self.env.cfg.perception.image_width
        camera_props.height = self.env.cfg.perception.image_height
        camera_props.horizontal_fov = self.env.cfg.perception.image_horizontal_fov


        self.cams = []

        for env_id in env_ids:

            cam = self.env.gym.create_camera_sensor(self.env.envs[env_id], camera_props)
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(camera_pose[0], camera_pose[1], camera_pose[2])
            quat_pitch = quat_from_angle_axis(torch.Tensor([-camera_rpy[1]]), torch.Tensor([0, 1, 0]))[0]
            quat_yaw = quat_from_angle_axis(torch.Tensor([camera_rpy[2]]), torch.Tensor([0, 0, 1]))[0]
            quat = quat_mul(quat_yaw, quat_pitch)
            local_transform.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            self.env.gym.attach_camera_to_body(cam, self.env.envs[env_id], 0,local_transform, gymapi.FOLLOW_TRANSFORM)
            self.cams.append(cam)
            # img = self.env.gym.get_camera_image_gpu_tensor(self.env.sim, self.env.envs[env_id], self.cams[env_id],
            #                                 gymapi.IMAGE_DEPTH)
            # img_tensor = gymtorch.wrap_tensor(img)
            # self.imgs.append(img_tensor)
            # initialize camera position
            # attach the camera to the base
            # trans_pos = gymapi.Vec3(camera_pose[0], camera_pose[1], camera_pose[2])
            # quat_pitch = quat_from_angle_axis(torch.Tensor([-camera_rpy[1]]), torch.Tensor([0, 1, 0]))[0]
            # quat_yaw = quat_from_angle_axis(torch.Tensor([camera_rpy[2]]), torch.Tensor([0, 0, 1]))[0]
            # quat = quat_mul(quat_yaw, quat_pitch)
            # trans_quat = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            # transform = gymapi.Transform(trans_pos, trans_quat)
            # follow_mode = gymapi.CameraFollowMode.FOLLOW_TRANSFORM
            # self.env.gym.attach_camera_to_body(cam, self.env.envs[env_id], 0, transform, follow_mode)

            

        return self.cams

    def get_observation(self, env_ids = None):

        raise NotImplementedError

    def plot_images(images, cols=2, ax_size=5, titles=None):
        # rows = (len(images)+cols-1)//cols
        rows = 1
        fig, axes = plt.subplots(rows, cols, figsize=(cols*ax_size, rows*ax_size))
        axes = axes.flatten()
        for ax in axes:
            ax.axis('off')
        for i in range(len(images)):
            axes[i].imshow(images[i])
            if titles:
                axes[i].set_title(titles[i], size=32)
        plt.tight_layout()

    def get_depth_images(self, env_ids = None):
        if env_ids is None: env_ids = range(self.env.num_envs)
        segmentation=True
        depth_images = []
        for env_id in env_ids:

            
            self.env.gym.start_access_image_tensors(self.env.sim)
            img = self.env.gym.get_camera_image_gpu_tensor(self.env.sim, self.env.envs[env_id], self.cams[env_id],
                                                                    gymapi.IMAGE_DEPTH)
            img_tensor = gymtorch.wrap_tensor(img)
            # img_tensor = self.imgs[env_id]
            
            img_tensor = torch.clamp(img_tensor,min = -10, max =-0.0)
            img_tensor = img_tensor*0.1
            # print (img_tensor)
            # if segmentation:
                # labels = cuda_slic(img_tensor.cpu().numpy(),n_segments=10, multichannel=False, compactness = 0.1)
                # slic = SlicAvx2(num_components=200, compactness=10,min_size_factor=0)
                # labels = slic.iterate(img_tensor.cpu().numpy().astype('uint8'))
                # labels = sk_slic(img_tensor.cpu().numpy(), n_segments=10, ,multichannel= False, max_num_iter=5)
                # marked = mark_boundaries(img_tensor.cpu().numpy(), labels)
                # import pdb;pdb.set_trace()
                # self.plot_images([img_tensor.cpu(), labels])
            # plt.imshow(img_tensor.cpu())
            # plt.pause(0.0000001)
            # # print(img_tensor)
            w, h = img.shape
            # print("training vision")
            self.env.gym.end_access_image_tensors(self.env.sim)
            

            depth_images.append((img_tensor.reshape([1, w, h])).to(self.env.device))
        depth_images = torch.cat(depth_images, dim=0)
        return depth_images
        # return img_tensor
    
    def get_rgb_images(self, env_ids):
        if env_ids is None: env_ids = range(self.env.num_envs)

        rgb_images = []
        for env_id in env_ids:
            img = self.env.gym.get_camera_image(self.env.sim, self.env.envs[env_id], self.cams[env_id],
                                            gymapi.IMAGE_COLOR)
            w, h = img.shape
            rgb_images.append(
                torch.from_numpy(img.reshape([1, w, h // 4, 4]).astype(np.int32)).to(self.env.device))
        rgb_images = torch.cat(rgb_images, dim=0)
        return rgb_images
    
    def get_segmentation_images(self, env_ids):
        if env_ids is None: env_ids = range(self.env.num_envs)

        segmentation_images = []
        for env_id in env_ids:
            self.env.gym.start_access_image_tensors(self.env.sim)
            img = self.env.gym.get_camera_image_gpu_tensor(self.env.sim, self.env.envs[env_id], self.cams[env_id],
                                            gymapi.IMAGE_SEGMENTATION)
            img_tensor = gymtorch.wrap_tensor(img)
            plt.imshow(img_tensor.cpu(),cmap='gray')
            plt.pause(0.0000001)
            w, h = img_tensor.shape
            self.env.gym.end_access_image_tensors(self.env.sim)
            segmentation_images.append(img_tensor.reshape([1, w, h]).to(self.env.device))
            

        segmentation_images = torch.cat(segmentation_images, dim=0)
        return segmentation_images