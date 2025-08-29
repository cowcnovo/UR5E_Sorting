# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .ur5e_config import UR5E_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class UR5ESortingEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 5.0

    sim: SimulationCfg = SimulationCfg(dt=1/100, render_interval=decimation)

    # robot(s)
    ur5e_cfg: ArticulationCfg = UR5E_CONFIG.replace(prim_path="/World/envs/env_.*/ur5e")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    # reward weights
    ee_pos_track_rew_weight = -7.0
    ee_pos_track_fg_rew_weight = 25.0
    ee_orient_track_rew_weight = 10.0
    lifting_rew_weight = 0.0
    ground_hit_avoidance_rew_weight = 2.0
    joint_2_tuning_rew_weight = 3.0
    gripper_rew_weight = 10.0
    object_moved_rew_weight = -10.0
    joint_vel_rew_weight = -6e-4
    action_rate_rew_weight = -1e-4

    # max number of objects
    starting_num_of_objects_class = 5
    max_num_of_objects_class = 10
    class_names = ["A", "B"]

    # curriculum learning settings
    start_adding_objects_episode = 2
    adding_objects_episodes_interval = 1

    # camera settings
    camera_width = 1280
    camera_height = 960

    # spaces definition
    action_space = 7
    observation_space = {
        "robot_state": 8 + 8 + 1 + 3 #+ len(class_names) 
    }
    state_space = 0

    sim.physx.gpu_max_rigid_patch_count = 1024*1024

@configclass
class UR5ESortingEnvCfg_Play(UR5ESortingEnvCfg):
    max_num_of_objects_class = 10
    starting_num_of_objects_class = 1
    start_adding_objects_episode = 2
    adding_objects_episodes_interval = 1