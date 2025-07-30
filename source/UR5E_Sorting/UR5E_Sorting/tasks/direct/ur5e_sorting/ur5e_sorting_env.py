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
        
        self.tray_episode_initial_pos = self.tray.data.root_pos_w.clone()
        self.tray_episode_initial_quat = self.tray.data.root_quat_w.clone()

        self.number_of_visible_objects_class = 1
        self.indices_of_visible_objects_class = [0]
        self.time_steps = 0

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

        # Spawn cubes (class A)
        for i in range(self.cfg.max_num_of_objects_class):
            object_name = f"object{self.cfg.class_names[0]}{i}"
            cube_size = sample_uniform(
                lower=torch.tensor([0.03, 0.03, 0.03], device=self.device),
                upper=torch.tensor([0.06, 0.06, 0.06], device=self.device),
                size=(3,),
                device=self.device,
            ).tolist()
            object_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{object_name}",
                spawn=sim_utils.CuboidCfg(
                    size=tuple(cube_size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0), metallic=0.5),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 1.20)),
            )
            self.scene.rigid_objects[object_name] = RigidObject(cfg=object_cfg)

        # Spawn cylinders (class B)
        for i in range(self.cfg.max_num_of_objects_class):
            object_name = f"object{self.cfg.class_names[1]}{i}"
            random_radius = sample_uniform(
                lower=torch.tensor([0.02], device=self.device),
                upper=torch.tensor([0.03], device=self.device),
                size=(1,),
                device=self.device,
            ).item()
            random_height = sample_uniform(
                lower=torch.tensor([0.03], device=self.device),
                upper=torch.tensor([0.06], device=self.device),
                size=(1,),
                device=self.device,
            ).item()
            object_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{object_name}",
                spawn=sim_utils.CylinderCfg(
                    radius=random_radius,
                    height=random_height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.5),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 1.20)),
            )
            self.scene.rigid_objects[object_name] = RigidObject(cfg=object_cfg)

        # Object to track
        self.object = self.scene.rigid_objects[f"object{self.cfg.class_names[0]}{0}"]

        # Non-visible objects tray
        nonvisible_objects_tray = sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "models/tray.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            scale=(3.0, 3.0, 15.0),
        )
        nonvisible_objects_tray.func("/World/envs/env_.*/nonvisible_objects_tray", nonvisible_objects_tray, translation=(0.6, 0.0, 5.0), orientation=(0.707, 0.0, 0.0, 0.707))

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

        # increase time steps
        if self.time_steps == 0:
            # Print info about the training and the objects
            print(f"Episode duration: {self.max_episode_length} timesteps")
            print(f"Number of objects: {self.cfg.max_num_of_objects_class}")
            print(f"Timesteps to start adding objects: {self.cfg.start_adding_objects_timestep}")
            print(f"Timesteps interval for adding objects: {self.cfg.adding_objects_timesteps_interval}")
            print(f"Required number of timesteps to add all objects: {(self.cfg.max_num_of_objects_class - 1)*self.cfg.adding_objects_timesteps_interval + self.cfg.start_adding_objects_timestep}")
            
        self.time_steps += 1

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
            self.tray,
            self.tray_episode_initial_pos,
            self.tray_episode_initial_quat,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check if the object is dropped
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
        rand_samples_tray = sample_uniform(
            lower=torch.tensor([-0.03, -0.03, 0.0], device=self.device),
            upper=torch.tensor([0.03, 0.03, 0.0], device=self.device),
            size=(len(env_ids), 3),
            device=self.device,
        )
        positions = root_states[:, :3] + self.scene.env_origins[env_ids] + rand_samples_tray
        orientations = root_states[:, 3:7]
        velocities = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        self.tray.write_root_state_to_sim(torch.cat([positions, orientations, velocities], dim=-1), env_ids=env_ids)

        self.tray_episode_initial_pos = self.tray.data.root_pos_w.clone()
        self.tray_episode_initial_quat = self.tray.data.root_quat_w.clone()

        # Gradually increase the number of visible objects in the scene
        if self.time_steps >= self.cfg.start_adding_objects_timestep and (self.time_steps - self.cfg.start_adding_objects_timestep) % self.cfg.adding_objects_timesteps_interval == 0:
            self.number_of_visible_objects_class += 1
            if self.number_of_visible_objects_class > self.cfg.max_num_of_objects_class:
                self.number_of_visible_objects_class = self.cfg.max_num_of_objects_class
        
        # Randomize the selected visible objects
        # Create a list of size number_of_visible_objects with random indices between 0 and self.cfg.max_num_of_objects_class - 1, without repition
        self.indices_of_visible_objects_class = torch.randperm(self.cfg.max_num_of_objects_class, device=self.device)[:self.number_of_visible_objects_class].tolist()

        # Randomize the object to track
        random_class = self.cfg.class_names[torch.randint(0, len(self.cfg.class_names), (1,)).item()]
        random_index = self.indices_of_visible_objects_class[torch.randint(0, len(self.indices_of_visible_objects_class), (1,)).item()]
        self.object = self.scene.rigid_objects[f"object{random_class}{random_index}"]

        # Randomize all objects
        for class_name in self.cfg.class_names:
            for i in range(self.cfg.max_num_of_objects_class):
                object_name = f"object{class_name}{i}"
                obj = self.scene.rigid_objects[object_name]

                object_is_visible = True if i in self.indices_of_visible_objects_class else False
                z_increment_for_visibility = 0.0 if object_is_visible else 4.2

                # Randomize object pose
                root_states = obj.data.default_root_state[env_ids].clone()
                rand_samples_object = sample_uniform(
                    lower=torch.tensor([-0.175, -0.25, z_increment_for_visibility], device=self.device),
                    upper=torch.tensor([0.175, 0.25, z_increment_for_visibility + 0.25], device=self.device),
                    size=(len(env_ids), 3),
                    device=self.device,
                )
                positions = root_states[:, :3] + self.scene.env_origins[env_ids] + rand_samples_object + rand_samples_tray
                orientations = root_states[:, 3:7]
                velocities = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
                obj.write_root_state_to_sim(torch.cat([positions, orientations, velocities], dim=-1), env_ids=env_ids)

        # Print info
        print("\nNumber of visible objects:", self.number_of_visible_objects_class)
        print("Indices of visible objects:", self.indices_of_visible_objects_class)
        print("Object to track:", self.object.cfg.prim_path)

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
    tray: RigidObject,
    tray_episode_initial_pos: torch.Tensor,
    tray_episode_initial_quat: torch.Tensor,
):
    ee_pos_track_rew = ee_pos_track_rew_weight * object_position_error(object, ee_frame)
    ee_pos_track_fg_rew = ee_pos_track_fg_rew_weight * object_position_error_tanh(object, ee_frame, std=0.1)
    ee_orient_track_rew = ee_orient_track_rew_weight * end_effector_orientation_error(ee_frame)
    lifting_rew = lifting_rew_weight * object_is_lifted(object, ee_frame, std=0.1, std_height=0.1, desired_height=1.3)
    ground_hit_avoidance_rew = ground_hit_avoidance_rew_weight * ground_hit_avoidance(object, ee_frame)
    joint_2_tuning_rew = joint_2_tuning_rew_weight * joint_2_tuning(ur5e_joint_pos)
    tray_moved_rew = tray_moved_rew_weight * tray_moved(tray, tray_episode_initial_pos, tray_episode_initial_quat, std=0.1)
    
    total_reward = ee_pos_track_rew + ee_pos_track_fg_rew + ee_orient_track_rew + lifting_rew + ground_hit_avoidance_rew + joint_2_tuning_rew + tray_moved_rew
    return total_reward