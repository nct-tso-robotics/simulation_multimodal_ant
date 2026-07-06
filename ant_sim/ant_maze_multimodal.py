"""Multimodal Ant maze environment adapted from jayLEE0301/vq_bet_official.

The environment is the AntMazeMultimodalEvalEnv from VQ-BeT's envs/antenv,
ported from gym/mujoco_py to gymnasium and the modern mujoco bindings. An ant
starts near the maze center and four goals sit at fixed corners; an episode
ends when all four goals are visited or after ``max_step`` steps. The order
in which goals are reached is the multimodality signal.
"""

import logging
import os
from copy import deepcopy

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box

QPOS_DIMENSION = 15
QVEL_DIMENSION = 14
ACTION_DIMENSION = 8
NUM_GOALS = 4
GOAL_DISTANCE_THRESHOLD = 1.5


class AntMazeMultimodalEvalEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Ant maze with four corner goals and multimodal completion tracking."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 10,
    }

    xml_filename = "ant_maze_multimodal.xml"
    mujoco_xml_full_path = os.path.join(
        os.path.dirname(__file__), "assets", xml_filename
    )
    reward_type = "sparse"
    distance_threshold = 0.5
    max_step = 1200
    height = 1000
    width = 1000

    def __init__(
        self,
        seed=0,
        render_mode="rgb_array",
    ):
        self.rng = np.random.RandomState(seed)
        self.nb_step = 0
        self.goal_cond = False
        self.achieved = np.array([0, 0, 0, 0])
        self.init_xy = np.array([0, 0])
        self.goal_arr = np.random.uniform(low=-4.0, high=20.0, size=(NUM_GOALS, 2))
        self.completion_ids = []

        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(QPOS_DIMENSION + QVEL_DIMENSION,),
            dtype=np.float32,
        )

        mujoco_env.MujocoEnv.__init__(
            self,
            model_path=self.mujoco_xml_full_path,
            frame_skip=5,
            observation_space=observation_space,
            render_mode=render_mode,
            default_camera_config={
                "lookat": np.array([8, 8, 0]),
                "distance": 30,
                "azimuth": 0,
                "elevation": -90,
            },
            height=self.height,
            width=self.width,
        )
        utils.EzPickle.__init__(self)
        self._check_model_parameter_dimensions()

    def _check_model_parameter_dimensions(self):
        if self.model.nq != QPOS_DIMENSION:
            raise ValueError(f"Expected nq={QPOS_DIMENSION}, got {self.model.nq}")
        if self.model.nv != QVEL_DIMENSION:
            raise ValueError(f"Expected nv={QVEL_DIMENSION}, got {self.model.nv}")
        if self.model.nu != ACTION_DIMENSION:
            raise ValueError(f"Expected nu={ACTION_DIMENSION}, got {self.model.nu}")

    def set_goalcond(self):
        self.goal_cond = True

    def step(self, a):
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        reward = 0
        for goal_idx in range(NUM_GOALS):
            goal = self.goal_arr[goal_idx]
            if self.goal_distance(ob["achieved_goal"], goal) <= (
                GOAL_DISTANCE_THRESHOLD
            ):
                if self.achieved[goal_idx] == 0:
                    self.achieved[goal_idx] = 1
                    logging.info(f"Achieved goal {goal_idx}")
                    self.completion_ids.append(goal_idx)
                    ob["goal_arr"] = deepcopy(
                        np.concatenate((self.goal_arr.flatten(), self.achieved))
                    )
        self.nb_step = 1 + self.nb_step
        done = bool(
            (self.nb_step > self.max_step) or np.sum(self.achieved) == NUM_GOALS
        )

        info = {
            "all_completions_ids": self.completion_ids,
        }

        return ob, reward, done, info

    def compute_reward(self, achieved_goal, goal, info=None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = np.array(dist) > self.distance_threshold
            return -rs.astype(np.float32)
        else:
            return -dist

    def _get_obs(self):
        obs = np.concatenate(
            [
                self.data.qpos.flat[:QPOS_DIMENSION],
                self.data.qvel.flat[:QVEL_DIMENSION],
            ]
        )
        achieved_goal = obs[:2]
        return {
            "observation": obs.copy(),
            "achieved_goal": deepcopy(achieved_goal),
            "desired_goal": obs.copy(),
            "goal_arr": deepcopy(
                np.concatenate((self.goal_arr.flatten(), self.achieved))
            ),
        }

    def set_task_goal(self, goal=None):
        if goal is not None:
            self.goal_arr = np.reshape(goal[29:37], (NUM_GOALS, 2))

    def set_achieved(self, one_indices):
        if self.goal_cond:
            self.achieved[one_indices] = 1

    def reset_model(self):
        self.goal_arr = np.random.uniform(low=-4.0, high=20.0, size=(NUM_GOALS, 2))
        self.goal_arr[0] = np.array([0.0, 0.0])
        self.goal_arr[1] = np.array([16.0, 0.0])
        self.goal_arr[2] = np.array([0.0, 16.0])
        self.goal_arr[3] = np.array([16.0, 16.0])
        self.achieved = np.array([0, 0, 0, 0])
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.rng.randn(self.model.nv) * 0.1
        self.init_xy = self.rng.uniform(low=7.0, high=9.0, size=2)
        self.init_qpos[:2] = self.init_xy
        qpos[:2] = self.init_xy

        qpos[QPOS_DIMENSION:] = self.init_qpos[QPOS_DIMENSION:]
        qvel[QVEL_DIMENSION:] = 0.0
        self.set_state(qpos, qvel)
        self.nb_step = 0
        self.completion_ids = []

        return self._get_obs()

    @property
    def completed_tasks(self):
        return [f"{idx}" for idx in self.completion_ids]

    def goal_distance(self, achieved_goal, goal):
        if achieved_goal.ndim == 1:
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist
