# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import numpy as np
import torch
import copy
from modules.real3d.segformer import SegFormerImg2PlaneBackbone
from modules.img2plane.triplane import OSGDecoder
from modules.eg3ds.models.superresolution import SuperresolutionHybrid8XDC
from modules.eg3ds.volumetric_rendering.renderer import ImportanceRenderer
from modules.eg3ds.volumetric_rendering.ray_sampler import RaySampler
from modules.img2plane.img2plane_model import Img2PlaneModel

from utils2.commons.hparams import hparams
import torch.nn.functional as F
import torch.nn as nn
from modules.real3d.facev2v_warp.layers import *
from einops import rearrange


class SameBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size=3, padding=1):
        super(SameBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding, padding_mode='replicate')
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding, padding_mode='replicate')
        self.norm1 = nn.GroupNorm(4, in_features, affine=True)
        self.norm2 = nn.GroupNorm(4, in_features, affine=True)
        self.alpha = nn.Parameter(torch.tensor([0.01]))

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = x + self.alpha * out
        return out


class Plane2GridModule(nn.Module):
    def __init__(self, triplane_depth=3, in_out_dim=96):
        super().__init__()
        self.triplane_depth = triplane_depth
        self.in_out_dim = in_out_dim
        if self.triplane_depth <= 3:
            self.num_layers_per_block = 1
        else:
            self.num_layers_per_block = 2
        self.res_blocks_3d = nn.Sequential(*[SameBlock3d(in_out_dim//3) for _ in range(self.num_layers_per_block)])
        
    def forward(self, x):
        x_inp = x # [1, 96*D, H, W]
        N, KCD, H, W = x.shape
        K, C, D = 3, KCD // self.triplane_depth // 3, self.triplane_depth
        assert C == self.in_out_dim // 3
        x = rearrange(x, 'n (k c d) h w -> (n k) c d h w', k=K, c=C, d=D) # ==> [1, 96, D, H, W]
        x = self.res_blocks_3d(x) # ==> [1, 96, D, H, W]
        x = rearrange(x, '(n k) c d h w -> n (k c d) h w', k=K)
        return x


class OSAvatar_Img2plane(torch.nn.Module):
    def __init__(self, hp=None):
        super().__init__()
        global hparams
        self.hparams = copy.copy(hparams) if hp is None else copy.copy(hp)
        hparams = self.hparams
        print("hparams",hparams)
        if hparams ==  {}:
            hparams = {'accumulate_grad_batches': 1, 'add_ffhq_singe_disc': False, 'also_update_decoder': False, 'amp': False, 'appearance_feat_mul_torso_mask': True, 'base_channel': 32768, 'base_config': ['./secc_img2plane_orig.yaml'], 'batch_size': 1, 'binary_data_dir': 'data/binary/CelebV-HQ', 'blur_fade_kimg': 20, 'blur_init_sigma': 10, 'blur_raw_target': True, 'box_warp': 1, 'ckpt_milestone_interval': 50000, 'clip_grad_norm': 1.0, 'clip_grad_value': 0, 'cond_hid_dim': 32, 'cond_out_dim': 16, 'cond_type': 'idexp_lm3d_normalized', 'debug': False, 'density_reg_p_dist': 0.004, 'disable_highreso_at_stage1': True, 'disc_c_noise': 1.0, 'disc_cond_mode': 'none', 'ds_name': 'FULL_Concat_VFHQ_CelebVHQ_TH1KH_RAVDESS', 'ema_interval': 400, 'enable_rescale_plane_regulation': False, 'eval_max_batches': 100, 'ffhq_disc_inp_mode': 'eg3d_gen', 'final_resolution': 512, 'flipped_to_world_coord': True, 'fuse_with_deform_source': False, 'gen_cond_mode': 'none', 'generator_condition_on_pose': True, 'gpc_reg_fade_kimg': 1000, 'gpc_reg_prob': 0.8, 'group_size_for_mini_batch_std': 2, 'htbsr_head_threshold': 0.9, 'htbsr_head_weight_fuse_mode': 'v2', 'img2plane_backbone_mode': 'composite', 'img2plane_backbone_scale': 'standard', 'init_from_ckpt': 'checkpoints/240207_robust_secc2plane/secc2plane_orig_blink0.3_pertubeNone/model_ckpt_steps_150000.ckpt', 'lam_occlusion_2_reg_l1': 0.0, 'lam_occlusion_reg_l1': 0.0, 'lam_occlusion_weights_entropy': 0.001, 'lambda_G_adversarial_adv': 1.0, 'lambda_G_supervise_adv': 1.0, 'lambda_G_supervise_mse': 1.0, 'lambda_G_supervise_mse_raw': 1.0, 'lambda_density_reg': 0.25, 'lambda_ffhq_mv_adv': 0.002, 'lambda_gradient_penalty': 1.0, 'lambda_mse': 1.0, 'lambda_mse_depth': 0.0, 'lambda_th1kh_mv_adv': 0.003, 'lambda_weights_entropy': 0.01, 'lambda_weights_l1': 0.1, 'load_ckpt': '', 'lpips_mode': 'vgg19_v2', 'lr_d': 0.0002, 'lr_decay_interval': 5000, 'lr_decay_rate': 0.95, 'lr_g': 1e-05, 'lr_lambda_pertube_secc': 0.01, 'lr_mul_cano_img2plane': 1.0, 'mapping_network_depth': 2, 'max_channel': 512, 'max_updates': 100000, 'mimic_plane': False, 'min_rescale_factor': 0.25, 'motion_smo_win_size': 5, 'neural_rendering_resolution': 128, 'normalize_cond': False, 'normalize_radius': False, 'not_save_modules': ['criterion_lpips', 'eg3d_model'], 'num_ckpt_keep': 1, 'num_fp16_layers_in_discriminator': 4, 'num_fp16_layers_in_generator': 0, 'num_fp16_layers_in_super_resolution': 4, 'num_samples_coarse': 48, 'num_samples_fine': 48, 'num_sanity_val_steps': 1, 'num_valid_plots': 25, 'num_workers': 8, 'ones_ws_for_sr': True, 'optimizer_adam_beta1_d': 0.0, 'optimizer_adam_beta1_g': 0.0, 'optimizer_adam_beta2_d': 0.99, 'optimizer_adam_beta2_g': 0.99, 'phase1_plane_fusion_mode': 'add', 'pncc_cond_mode': 'cano_src_tgt', 'pretrained_eg3d_ckpt': '/home/tiger/projects/GeneFace_private/checkpoints/geneface2_ckpts/eg3d_baseline_run2/model_ckpt_steps_100000.ckpt', 'print_nan_grads': False, 'process_id': 0, 'processed_data_dir': 'data/processed/videos', 'random_sample_pose': True, 'raw_data_dir': '/home/tiger/datasets/raw/FFHQ', 'ray_far': 'auto', 'ray_near': 'auto', 'reg_interval_d': 16, 'reg_interval_g': 4, 'reg_interval_g_cond': 4, 'reload_head_ckpt': '', 'resume_from_checkpoint': 0, 'save_best': True, 'save_codes': ['tasks', 'modules', 'egs'], 'secc_pertube_mode': 'randn', 'secc_pertube_randn_scale': 0.01, 'secc_segformer_scale': 'b0', 'seed': 9999, 'seg_out_mode': 'head', 'smo_win_size': 5, 'split_seed': 999, 'sr_type': 'vanilla', 'start_adv_iters': 40000, 'target_pertube_blink_secc_loss': 0.15, 'target_pertube_secc_loss': 0.5, 'task_cls': 'tasks.os_avatar.secc_img2plane_torso_task.SECC_Img2PlaneEG3D_TorsoTask', 'tb_log_interval': 100, 'torch_compile': True, 'torso_inp_mode': 'rgb_alpha', 'torso_kp_num': 4, 'torso_model_version': 'v2', 'torso_occlusion_reg_unmask_factor': 0.3, 'torso_ref_segout_mode': 'torso', 'total_process': 1, 'triplane_depth': 1, 'triplane_feature_type': 'triplane', 'triplane_hid_dim': 32, 'two_stage_training': True, 'update_on_th1kh_samples': False, 'update_src2src_interval': 16, 'use_kv_dataset': True, 'use_motion_smo_net': False, 'use_mse': False, 'use_th1kh_disc': False, 'use_th1kh_mv_adv': False, 'val_check_interval': 2000, 'valid_infer_interval': 2000, 'valid_monitor_key': 'val_loss', 'valid_monitor_mode': 'min', 'video_id': 'May', 'w_dim': 512, 'warmup_updates': 4000, 'weight_fuse': True, 'work_dir': '', 'z_dim': 512, 'infer': False, 'validate': False, 'exp_name': '', 'start_rank': 0, 'world_size': -1, 'init_method': 'tcp'}

        self.camera_dim = 25 # extrinsic 4x4 + intrinsic 3x3
        self.neural_rendering_resolution = hparams.get("neural_rendering_resolution", 128)
        self.w_dim = hparams['w_dim']
        self.img_resolution = hparams['final_resolution']
        self.triplane_depth = hparams.get("triplane_depth", 1)
        
        self.triplane_hid_dim = triplane_hid_dim = hparams.get("triplane_hid_dim", 32)
        # extract canonical triplane from src img
        self.img2plane_backbone = Img2PlaneModel(out_channels=3*triplane_hid_dim*self.triplane_depth, hp=hparams)
        if hparams.get("triplane_feature_type", "triplane") in ['trigrid_v2']:
            self.plane2grid_module = Plane2GridModule(triplane_depth=self.triplane_depth, in_out_dim=3*triplane_hid_dim) # add depth here
          
        # positional embedding
        self.decoder = OSGDecoder(triplane_hid_dim, {'decoder_lr_mul': 1, 'decoder_output_dim': triplane_hid_dim})
        # create super resolution network
        self.sr_num_fp16_res = 0
        self.sr_kwargs = {'channel_base': hparams['base_channel'], 'channel_max': hparams['max_channel'], 'fused_modconv_default': 'inference_only'}
        self.superresolution = SuperresolutionHybrid8XDC(channels=triplane_hid_dim, img_resolution=self.img_resolution, sr_num_fp16_res=self.sr_num_fp16_res, sr_antialias=True, large_sr=hparams.get('large_sr',False), **self.sr_kwargs)
        # Rendering Options
        self.renderer = ImportanceRenderer(hp=hparams)
        self.ray_sampler = RaySampler()
        self.rendering_kwargs = {'image_resolution': hparams['final_resolution'], 
                            'disparity_space_sampling': False, 
                            'clamp_mode': 'softplus',
                            'gpc_reg_prob': hparams['gpc_reg_prob'], 
                            'c_scale': 1.0, 
                            'superresolution_noise_mode': 'none', 
                            'density_reg': hparams['lambda_density_reg'], 'density_reg_p_dist': hparams['density_reg_p_dist'], 
                            'reg_type': 'l1', 'decoder_lr_mul': 1.0, 
                            'sr_antialias': True, 
                            'depth_resolution': hparams['num_samples_coarse'], 
                            'depth_resolution_importance': hparams['num_samples_fine'],
                            'ray_start': 'auto', 'ray_end': 'auto',
                            'box_warp': hparams.get("box_warp", 1.), # 3DMM坐标系==world坐标系，而3DMM的landmark的坐标均位于[-1,1]内
                            'avg_camera_radius': 2.7,
                            'avg_camera_pivot': [0, 0, 0.2],
                            'white_back': False,
                            }

    def cal_plane(self, img, cond=None, ret=None, **synthesis_kwargs):
        hparams = self.hparams
        planes = self.img2plane_backbone(img, cond, **synthesis_kwargs) #  [B, 3, C*D, H, W]
        if hparams.get("triplane_feature_type", "triplane") in ['triplane', 'trigrid']:
            planes = planes.view(len(planes), 3, self.triplane_hid_dim*self.triplane_depth, planes.shape[-2], planes.shape[-1])
        elif hparams.get("triplane_feature_type", "triplane") in ['trigrid_v2']:
            b, k, cd, h, w = planes.shape
            planes = planes.reshape([b, k*cd, h, w])
            planes = self.plane2grid_module(planes)
            planes = planes.reshape([b, k, cd, h, w])
        else:
            raise NotImplementedError()
        return planes # [B, 3, C*D, H, W]
    
    def _forward_sr(self, rgb_image, feature_image, cond, ret, **synthesis_kwargs):
        hparams = self.hparams
        ones_ws = torch.ones([feature_image.shape[0], 14, hparams['w_dim']], dtype=feature_image.dtype, device=feature_image.device)
        if hparams.get("sr_type", "vanilla") == 'vanilla':
            sr_image = self.superresolution(rgb_image, feature_image, ones_ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        elif hparams.get("sr_type", "vanilla") == 'spade':
            sr_image = self.superresolution(rgb_image, feature_image, ones_ws, segmap=cond['ref_head_img'], noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        return sr_image

    def synthesis(self, img, camera, cond=None, ret=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        hparams = self.hparams
        if ret is None: ret = {}
        cam2world_matrix = camera[:, :16].view(-1, 4, 4)
        intrinsics = camera[:, 16:25].view(-1, 3, 3)

        neural_rendering_resolution = self.neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.cal_plane(img, cond, ret, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes
        
        # Perform volume rendering
        feature_samples, depth_samples, weights_samples, is_ray_valid = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        weights_image = weights_samples.permute(0, 2, 1).reshape(N,1,H,W).contiguous() # [N,1,H,W]
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        if hparams.get("mask_invalid_rays", False):
            is_ray_valid_mask = is_ray_valid.reshape([feature_samples.shape[0], 1,self.neural_rendering_resolution,self.neural_rendering_resolution]) # [B, 1, H, W]
            feature_image[~is_ray_valid_mask.repeat([1,feature_image.shape[1],1,1])] = -1
            # feature_image[~is_ray_valid_mask.repeat([1,feature_image.shape[1],1,1])] *= 0
            # feature_image[~is_ray_valid_mask.repeat([1,feature_image.shape[1],1,1])] -= 1
            depth_image[~is_ray_valid_mask] = depth_image[is_ray_valid_mask].min().item()

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        ret['weights_img'] = weights_image
        sr_image = self._forward_sr(rgb_image, feature_image, cond, ret, **synthesis_kwargs)
        rgb_image = rgb_image.clamp(-1,1)
        sr_image = sr_image.clamp(-1,1)
        ret.update({'image_raw': rgb_image, 'image_depth': depth_image, 'image': sr_image, 'image_feature': feature_image[:, 3:], 'plane': planes})
        return ret

    def sample(self, coordinates, directions, img, cond=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, ref_camera=None, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        planes = self.cal_plane(img, cond, ret={}, ref_camera=ref_camera)
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, img, camera, cond=None, ret=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, return_all=True, **synthesis_kwargs):
        # Render a batch of generated images.
        out = self.synthesis(img, camera, cond=cond, ret=ret, update_emas=update_emas, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)

        return out
