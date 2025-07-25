# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, FrameTransformer
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg, TiledCamera
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms

from .ur5e_sorting_env_cfg import UR5ESortingEnvCfg


from .mdp.rewards import object_position_error, object_position_error_tanh, end_effector_orientation_error
from .mdp.rewards import object_is_lifted, ground_hit_avoidance, joint_2_tuning, tray_moved


class UR5ESortingEnv(DirectRLEnv):
    cfg: UR5ESortingEnvCfg

    def __init__(self, cfg: UR5ESortingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.arm_joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.gripper_joint_names=["left_outer_knuckle_joint", "right_outer_knuckle_joint"]
        self.arm_joints_ids, _ = self.ur5e.find_joints(name_keys=self.arm_joint_names) # returns ids, names
        self.gripper_joints_ids, _ = self.ur5e.find_joints(name_keys=self.gripper_joint_names) # returns ids, names

        self.ur5e_joint_pos = self.ur5e.data.joint_pos # all 12 joints, inlcuding unactuated ones
        self.ur5e_joint_vel = self.ur5e.data.joint_vel

    def _setup_scene(self):

        # Ground-plane
        ground_cfg = sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "models/plane.usd"),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.5),
            scale=(1000.0, 1000.0, 1.0),
        )
        ground_cfg.func("/World/GroundPlane", ground_cfg)

        # lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1200.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/light", light_cfg)

        # Clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # spawn a usd file of a table into the scene
        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        )
        table_cfg.func("/World/envs/env_.*/table", table_cfg, translation=(0.6, 0.0, 1.10), orientation=(0.707, 0.0, 0.0, 0.707))

        # Tray
        tray_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/tray",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(os.path.dirname(__file__), "models/tray.usd"),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 1.15), rot=(0.707, 0.0, 0.0, 0.707)),
        )
        self.tray = RigidObject(cfg=tray_cfg)
        self.scene.rigid_objects["tray"] = self.tray

        # spawn a cuboid with colliders and rigid body
        object_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/object",
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0), metallic=0.5),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 1.25)),
        )
        self.object = RigidObject(cfg=object_cfg)
        self.scene.rigid_objects["object"] = self.object

        # robot
        self.ur5e = Articulation(self.cfg.ur5e_cfg)
        self.scene.articulations["ur5e"] = self.ur5e

        # end-effector frame
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/ur5e/base",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/ur5e/gripper_end_effector",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )
        self.ee_frame = FrameTransformer(cfg=ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self.ee_frame # without this, the frame will not be updated

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Separate arm actions and gripper actions
        arm_actions = self.actions[:, :-1]  
        gripper_action = self.actions[:, -1].unsqueeze(-1)

        # Apply arm actions
        self.ur5e.set_joint_position_target(arm_actions, joint_ids=self.arm_joints_ids)

        # Apply gripper actions (binary control)
        gripper_joint_positions = torch.where(
            gripper_action < 0,  # Threshold for binary control
            torch.tensor([0.698, -0.698], device=gripper_action.device),  # Closed position
            torch.tensor([0.0, 0.0], device=gripper_action.device),  # Open position
        )
        self.ur5e.set_joint_position_target(gripper_joint_positions, joint_ids=self.gripper_joints_ids)

    def _get_observations(self) -> dict:
        # Determine object position in the robot's base frame
        object_pos_w = self.object.data.root_pos_w[:, :3]  # Object position in world frame
        robot_pos_w = self.ur5e.data.root_state_w[:, :3]  # Robot base position in world frame
        robot_quat_w = self.ur5e.data.root_state_w[:, 3:7]  # Robot base orientation in world frame
        object_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, object_pos_w)

        # Concatenate robot state and object position for observations
        robot_state = torch.cat(
            [
                self.ur5e_joint_pos[:, self.arm_joints_ids],
                self.ur5e_joint_vel[:, self.arm_joints_ids],
                self.ur5e_joint_pos[:, self.gripper_joints_ids],
                self.ur5e_joint_vel[:, self.gripper_joints_ids],
                object_pos_b
            ],
            dim=-1,
        )

        observations = {
            "policy": {
                "robot_state": robot_state,
            }
        }

        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.ee_pos_track_rew_weight,
            self.cfg.ee_pos_track_fg_rew_weight,
            self.cfg.ee_orient_track_rew_weight,
            self.cfg.lifting_rew_weight,
            self.cfg.ground_hit_avoidance_rew_weight,
            self.cfg.joint_2_tuning_rew_weight,
            self.cfg.tray_moved_rew_weight,
            self.object,
            self.ee_frame,
            self.ur5e_joint_pos,
            self.tray
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        object_height = self.object.data.root_pos_w[:, 2]
        object_dropped = object_height < 0.8

        # Check if the episode has timed out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return object_dropped, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.ur5e._ALL_INDICES
        super()._reset_idx(env_ids) # without this, the episode length buffer will not be reset

        # Reset robot joints to default positions and velocities
        joint_pos = self.ur5e.data.default_joint_pos[env_ids] + sample_uniform(
            lower=-0.15,
            upper=0.15,
            size=(len(env_ids), self.ur5e.num_joints),
            device=self.device,
        )
        joint_vel = self.ur5e.data.default_joint_vel[env_ids]
        self.ur5e_joint_pos[env_ids] = joint_pos
        self.ur5e_joint_vel[env_ids] = joint_vel
        self.ur5e.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset tray
        root_states = self.tray.data.default_root_state[env_ids].clone()
        rand_samples = sample_uniform(
            lower=torch.tensor([-0.0, -0.0, 0.0], device=self.device),
            upper=torch.tensor([0.0, 0.0, 0.0], device=self.device),
            size=(len(env_ids), 3),
            device=self.device,
        )
        positions = root_states[:, :3] + self.scene.env_origins[env_ids] + rand_samples
        orientations = root_states[:, 3:7]
        velocities = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        self.tray.write_root_state_to_sim(torch.cat([positions, orientations, velocities], dim=-1), env_ids=env_ids)

        # Randomize object position
        root_states = self.object.data.default_root_state[env_ids].clone()
        rand_samples = sample_uniform(
            lower=torch.tensor([-0.15, -0.2, 0.0], device=self.device),
            upper=torch.tensor([0.15, 0.2, 0.0], device=self.device),
            size=(len(env_ids), 3),
            device=self.device,
        )
        positions = root_states[:, :3] + self.scene.env_origins[env_ids] + rand_samples
        orientations = root_states[:, 3:7]
        velocities = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        self.object.write_root_state_to_sim(torch.cat([positions, orientations, velocities], dim=-1), env_ids=env_ids)


#@torch.jit.script
def compute_rewards(
    ee_pos_track_rew_weight: float,
    ee_pos_track_fg_rew_weight: float,
    ee_orient_track_rew_weight: float,
    lifting_rew_weight: float,
    ground_hit_avoidance_rew_weight: float,
    joint_2_tuning_rew_weight: float,
    tray_moved_rew_weight: float,
    object: RigidObject,
    ee_frame: FrameTransformer,
    ur5e_joint_pos: torch.Tensor,
    tray: RigidObject
):
    ee_pos_track_rew = ee_pos_track_rew_weight * object_position_error(object, ee_frame)
    ee_pos_track_fg_rew = ee_pos_track_fg_rew_weight * object_position_error_tanh(object, ee_frame, std=0.1)
    ee_orient_track_rew = ee_orient_track_rew_weight * end_effector_orientation_error(ee_frame)
    lifting_rew = lifting_rew_weight * object_is_lifted(object, ee_frame, std=0.1, std_height=0.1)
    ground_hit_avoidance_rew = ground_hit_avoidance_rew_weight * ground_hit_avoidance(object, ee_frame)
    joint_2_tuning_rew = joint_2_tuning_rew_weight * joint_2_tuning(ur5e_joint_pos)
    tray_moved_rew = tray_moved_rew_weight * tray_moved(tray)
    
    total_reward = ee_pos_track_rew + ee_pos_track_fg_rew + ee_orient_track_rew + lifting_rew + ground_hit_avoidance_rew + joint_2_tuning_rew + tray_moved_rew
    return total_reward