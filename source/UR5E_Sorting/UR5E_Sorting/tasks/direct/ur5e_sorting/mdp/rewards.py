from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.utils.math import quat_error_magnitude, quat_mul

def tray_moved(tray: RigidObject, tray_episode_initial_pos: torch.Tensor, tray_episode_initial_quat: torch.Tensor, std: float) -> torch.Tensor: 
    tray_root_pos = tray.data.root_pos_w
    tray_root_quat = tray.data.root_quat_w

    tray_pos_diff = torch.norm(tray_root_pos - tray_episode_initial_pos, dim=-1)
    tray_quat_diff = quat_error_magnitude(tray_root_quat, tray_episode_initial_quat)

    reward = torch.tanh((tray_pos_diff + tray_quat_diff) / std)

    return reward

def joint_2_tuning(ur5e_joint_pos: torch.Tensor) -> torch.Tensor:
    joint_2_pos = ur5e_joint_pos[:, 1]  # Joint 2 position
    reward = torch.tanh(-joint_2_pos / 0.5) 
    
    return reward

def end_effector_orientation_error(ee_frame: FrameTransformer) -> torch.Tensor:
    number_of_envs = ee_frame.data.target_quat_w.shape[0]

    des_quat_w = torch.tensor([0.0, 0.0, 1.0, 0.0], device=ee_frame.device).repeat(number_of_envs, 1)
    curr_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    return quat_error_magnitude(curr_quat_w, des_quat_w)

def object_position_error(tracking_object_positions: torch.Tensor, ee_frame: FrameTransformer) -> torch.Tensor:
    cube_pos_w = tracking_object_positions  # Object position in world frame
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # End-effector position in world frame

    object_ee_distance = torch.norm(ee_pos_w - cube_pos_w, dim=1)
    return object_ee_distance

def object_position_error_tanh(tracking_object_positions: torch.Tensor, ee_frame: FrameTransformer, std: float) -> torch.Tensor:
    object_ee_distance = object_position_error(tracking_object_positions, ee_frame)
    reward = 1 - torch.tanh(object_ee_distance / std)
    return reward

def object_is_lifted(tracking_object_positions: torch.Tensor, ee_frame: FrameTransformer, std: float, std_height: float, desired_height: float) -> torch.Tensor:
    object_height_from_desired = desired_height - tracking_object_positions[:, 2]
    object_height_reward = 1 - torch.tanh(object_height_from_desired / std_height)

    reach_reward = object_position_error_tanh(tracking_object_positions, ee_frame, std)
    reward = reach_reward * object_height_reward

    return reward

def ground_hit_avoidance(tracking_object_positions: torch.Tensor, ee_frame: FrameTransformer) -> torch.Tensor:
    cube_z_pos_w = tracking_object_positions[:, 2]
    ee_z_pos_w = ee_frame.data.target_pos_w[..., 0, 2]

    height = ee_z_pos_w - cube_z_pos_w
    reward = 0.5 * torch.tanh(1000 * (height + 0.003)) - 0.5

    return reward