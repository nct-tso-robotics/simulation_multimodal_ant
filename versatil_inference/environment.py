"""Multimodal Ant environment manager for VersatIL policy evaluation."""

import csv
import datetime
import gc
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
from tso_robotics_sockets import InferenceResponseKey, ServerStatus
from versatil_constants.multimodal_ant import MultimodalAntProprioKey

from ant_sim import AntMazeMultimodalEvalEnv
from versatil_inference.episode_recorder import EpisodeRecorder
from versatil_inference.socket_flags import (
    DEFAULT_CLIENT_NAME,
    MAX_STEPS,
    NO_OP_ACTION,
    AntTrajectoryColumn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

ANT_TASK_NAME = "multimodal_ant"
QPOS_DIMENSION = 15
QVEL_DIMENSION = 14
GOAL_COORDS_DIMENSION = 8
NUM_GOALS = 4


class Environment:
    """Manages batched multimodal Ant environments."""

    def __init__(
        self,
        seed: int,
        num_trials: int,
        output_folder: str,
        max_parallel_envs: int = 10,
        record_video: bool = False,
    ):
        self.seed = seed
        self.num_trials = num_trials
        self.num_envs = num_trials
        self.output_folder = output_folder
        self.max_parallel_envs = max_parallel_envs
        self.record_video = record_video
        self.current_status = ServerStatus.CREATING_ENV.value
        self.client_name = DEFAULT_CLIENT_NAME
        self._rollout_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._episode_seeds: list[int] = [seed + i for i in range(num_trials)]

        self.active_environments = [False] * num_trials
        self.steps_counts = [0] * num_trials
        self.number_of_resets = [0] * num_trials
        self.environments_goals_achieved = [0] * num_trials
        self.environments_behavior_order = ["none"] * num_trials
        self.recorders: list[EpisodeRecorder | None] = [None] * num_trials
        self.trajectory_columns = [col.value for col in AntTrajectoryColumn]

        self._batch_global_indices: list[int] = []
        self._batch_envs: list[AntMazeMultimodalEvalEnv] = []
        self.latest_observation: dict[int, dict] = {}
        self.recently_reset_indices: list[int] = []

        logging.info(
            f"Multimodal Ant evaluation: {num_trials} trials, "
            f"seeds=[{self._episode_seeds[0]}..{self._episode_seeds[-1]}], "
            f"max {max_parallel_envs} parallel, record_video={record_video}"
        )

    @property
    def rollout_directory(self) -> Path:
        client_path = Path(self.client_name)
        if self.output_folder:
            safe_client_name = self.client_name.strip("/").replace("/", "_")
            return (
                Path(self.output_folder)
                / safe_client_name
                / ANT_TASK_NAME
                / self._rollout_date
            )
        return (
            client_path.parent
            / "rollouts"
            / client_path.name
            / ANT_TASK_NAME
            / self._rollout_date
        )

    def initialize(self) -> None:
        batch_size = min(self.max_parallel_envs, self.num_envs)
        self._batch_global_indices = list(range(batch_size))
        self._create_batch_environments()
        self.current_status = ServerStatus.WAITING_ACTION.value

    def get_latest_observation(self) -> dict[int, dict]:
        return self.latest_observation

    def consume_reset_indices(self) -> list[int]:
        indices = self.recently_reset_indices
        self.recently_reset_indices = []
        return indices

    def step(self, actions: dict[int, list[float]]) -> None:
        if self.current_status == ServerStatus.FINISHED.value:
            return

        rollout_directory = self.rollout_directory
        self.recently_reset_indices = []
        new_latest_observation: dict[int, dict] = {}

        for local_index, global_index in enumerate(self._batch_global_indices):
            if not self.active_environments[global_index]:
                continue

            env = self._batch_envs[local_index]
            action = self._action_for_env(local_index, actions)
            gym_obs, reward, done, info = env.step(action)
            self.steps_counts[global_index] += 1
            step_count = self.steps_counts[global_index]

            full_obs = self._build_full_obs(gym_obs=gym_obs, step_count=step_count)
            frame = self._render_frame(env)
            self.recorders[global_index].add_observation(
                frame=frame,
                trajectory_row=self._build_trajectory_row(full_obs),
                reward=float(reward),
                output_directory=rollout_directory,
            )

            terminated = bool(done) or step_count >= MAX_STEPS
            if terminated:
                self._finalize_episode(
                    global_index=global_index,
                    step_count=step_count,
                    rollout_directory=rollout_directory,
                    completed_goal_ids=list(info.get("all_completions_ids", [])),
                )
            else:
                new_latest_observation[local_index] = full_obs

        self.latest_observation = new_latest_observation
        self._advance_status_after_step()

    def close(self) -> None:
        for env in self._batch_envs:
            try:
                env.close()
            except Exception:
                logging.exception("Failed to close Ant environment")
        self._batch_envs = []

    def _create_batch_environments(self) -> None:
        batch_size = len(self._batch_global_indices)
        logging.info(
            f"Creating batch: trials {self._batch_global_indices[0]}-"
            f"{self._batch_global_indices[-1]} ({batch_size} envs)"
        )
        self._batch_envs = []
        rollout_directory = self.rollout_directory
        new_observation: dict[int, dict] = {}

        for local_index, global_index in enumerate(self._batch_global_indices):
            episode_seed = self._episode_seeds[global_index]
            env = AntMazeMultimodalEvalEnv(
                seed=episode_seed,
                render_mode="rgb_array" if self.record_video else None,
            )
            gym_obs, _ = env.reset(seed=episode_seed)
            self._batch_envs.append(env)

            self.active_environments[global_index] = True
            self.steps_counts[global_index] = 0
            self.recorders[global_index] = EpisodeRecorder(
                environment_id=ANT_TASK_NAME,
                language_instruction=f"seed_{episode_seed}",
                trajectory_columns=self.trajectory_columns,
            )
            full_obs = self._build_full_obs(gym_obs=gym_obs, step_count=0)
            self.recorders[global_index].add_observation(
                frame=self._render_frame(env),
                trajectory_row=self._build_trajectory_row(full_obs),
                reward=0.0,
                output_directory=rollout_directory,
            )
            new_observation[local_index] = full_obs

        self.latest_observation = new_observation
        self.recently_reset_indices = list(range(batch_size))

    def _advance_to_next_batch(self) -> bool:
        self.close()
        gc.collect()
        next_start = self._batch_global_indices[-1] + 1
        if next_start >= self.num_envs:
            return False
        end = min(next_start + self.max_parallel_envs, self.num_envs)
        self._batch_global_indices = list(range(next_start, end))
        self._create_batch_environments()
        return True

    def _advance_status_after_step(self) -> None:
        batch_active = any(
            self.active_environments[gi] for gi in self._batch_global_indices
        )
        if batch_active:
            self.current_status = ServerStatus.WAITING_ACTION.value
            return
        if self._advance_to_next_batch():
            self.current_status = ServerStatus.WAITING_ACTION.value
            return
        self._write_results_csv()
        self.current_status = ServerStatus.FINISHED.value

    def _finalize_episode(
        self,
        global_index: int,
        step_count: int,
        rollout_directory: Path,
        completed_goal_ids: list[int],
    ) -> None:
        goals_achieved = len(set(completed_goal_ids))
        behavior_order = self._behavior_order(completed_goal_ids)

        self.environments_goals_achieved[global_index] = goals_achieved
        self.environments_behavior_order[global_index] = behavior_order
        self.number_of_resets[global_index] += 1

        logging.info(
            f"Trial {global_index} "
            f"(seed={self._episode_seeds[global_index]}): "
            f"done, goals={goals_achieved}/{NUM_GOALS}, "
            f"behavior_order={behavior_order}, steps={step_count}"
        )
        self.recorders[global_index].save(
            goals_achieved=goals_achieved,
            behavior_order=behavior_order,
            output_directory=rollout_directory,
        )
        self.active_environments[global_index] = False
        self.recorders[global_index] = None

    @staticmethod
    def _behavior_order(completed_goal_ids: list[int]) -> str:
        if not completed_goal_ids:
            return "none"
        return "->".join(str(goal_id) for goal_id in completed_goal_ids)

    def behavior_order_counts(self) -> Counter:
        return Counter(self.environments_behavior_order)

    def first_goal_counts(self) -> Counter:
        return Counter(
            order.split("->")[0]
            for order in self.environments_behavior_order
            if order != "none"
        )

    def first_goal_entropy(self) -> float:
        """Exponentiated Shannon entropy over the first goal reached.

        Measures how many of the four goals the policy actually commits to
        first across trials; 1.0 means a single mode, 4.0 a uniform spread.
        """
        counts = self.first_goal_counts()
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log(probability)
        return float(math.exp(entropy))

    def _action_for_env(
        self,
        local_index: int,
        actions: dict[int, list[float]],
    ) -> np.ndarray:
        if local_index in actions:
            return np.asarray(actions[local_index], dtype=np.float64).reshape(-1)[
                : len(NO_OP_ACTION)
            ]
        return np.asarray(NO_OP_ACTION, dtype=np.float64)

    def _render_frame(self, env: AntMazeMultimodalEvalEnv) -> np.ndarray | None:
        if not self.record_video:
            return None
        try:
            return env.render()
        except Exception:
            logging.exception("Ant render failed")
            return None

    def _build_full_obs(self, gym_obs: dict, step_count: int) -> dict:
        state = np.asarray(gym_obs["observation"], dtype=np.float32).reshape(-1)
        achieved = np.asarray(
            gym_obs["goal_arr"][GOAL_COORDS_DIMENSION:], dtype=np.float32
        )
        # The unconditional training data stores zeros in the goal-coordinate
        # dims, so the served goal coordinates are zeroed to match.
        goal_coords = np.zeros(GOAL_COORDS_DIMENSION, dtype=np.float32)
        return {
            MultimodalAntProprioKey.QPOS.value: state[:QPOS_DIMENSION],
            MultimodalAntProprioKey.QVEL.value: state[
                QPOS_DIMENSION : QPOS_DIMENSION + QVEL_DIMENSION
            ],
            MultimodalAntProprioKey.GOAL_COORDS.value: goal_coords,
            MultimodalAntProprioKey.ACHIEVED.value: achieved,
            InferenceResponseKey.TIMESTEP.value: step_count,
        }

    def _build_trajectory_row(self, full_obs: dict) -> dict[str, float]:
        qpos = full_obs[MultimodalAntProprioKey.QPOS.value]
        achieved = full_obs[MultimodalAntProprioKey.ACHIEVED.value]
        return {
            AntTrajectoryColumn.TORSO_X.value: float(qpos[0]),
            AntTrajectoryColumn.TORSO_Y.value: float(qpos[1]),
            AntTrajectoryColumn.GOALS_ACHIEVED.value: float(np.sum(achieved)),
        }

    def _write_results_csv(self) -> None:
        output_directory = self.rollout_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        csv_path = output_directory / "results.csv"

        total_trials = sum(self.number_of_resets)
        mean_goals = (
            sum(self.environments_goals_achieved) / total_trials
            if total_trials > 0
            else 0.0
        )
        entropy = self.first_goal_entropy()

        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["trial", "seed", "goals_achieved", "behavior_order"])
            for i in range(self.num_envs):
                writer.writerow(
                    [
                        i,
                        self._episode_seeds[i],
                        self.environments_goals_achieved[i],
                        self.environments_behavior_order[i],
                    ]
                )
            writer.writerow([])
            writer.writerow(
                [
                    "mean",
                    f"{total_trials}",
                    f"{mean_goals:.4f}",
                    f"first_goal_entropy={entropy:.4f}",
                ]
            )
            writer.writerow([])
            writer.writerow(["first_goal", "count"])
            for label, count in sorted(self.first_goal_counts().items()):
                writer.writerow([label, count])
        logging.info(f"Results saved to {csv_path}")
