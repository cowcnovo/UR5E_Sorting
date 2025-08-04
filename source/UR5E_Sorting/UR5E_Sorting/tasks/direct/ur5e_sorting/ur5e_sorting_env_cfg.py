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
    ee_pos_track_rew_weight = -3.0 # Negative
    ee_pos_track_fg_rew_weight = 20.0 # Positive
    ee_orient_track_rew_weight = -3.0 # Negative
    lifting_rew_weight = 75.0 # Positive
    ground_hit_avoidance_rew_weight = 0.0 # Positive
    joint_2_tuning_rew_weight = 2.0 # Negative
    tray_moved_rew_weight = -0.0 # Negative

    # max number of objects
    max_num_of_objects_class = 15
    class_names = ["A", "B"]

    # curriculum learning settings
    start_adding_objects_episode = 15
    adding_objects_episodes_interval = 4

    # spaces definition
    action_space = 7
    observation_space = {
        "robot_state": 19
    }
    state_space = 0

    sim.physx.gpu_max_rigid_patch_count = 1024*1024

    