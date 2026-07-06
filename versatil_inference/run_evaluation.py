"""Runs multimodal Ant evaluation with a VersatIL policy client."""

import datetime
import logging
import os
from dataclasses import dataclass

import draccus
import wandb
from tso_robotics_sockets import ServerStatus, TransportKey

from versatil_inference.server import MultimodalAntServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DATE_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class EvalConfig:
    """Configuration for multimodal Ant evaluation."""

    seed: int = 42
    num_trials: int = 50
    ip_address: str = "0.0.0.0"
    port: int = 5556
    compression_type: str = "raw"
    output_folder: str = ""
    max_parallel_envs: int = 10
    record_video: bool = False
    run_id_note: str | None = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "multimodal-ant-eval"
    wandb_entity: str = ""


def run_evaluation(config: EvalConfig) -> None:
    run_id = f"EVAL-multimodal-ant-{DATE_TIME}"
    if config.run_id_note:
        run_id += f"--{config.run_id_note}"
    os.makedirs(config.local_log_dir, exist_ok=True)
    if config.use_wandb:
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=run_id,
        )

    server = MultimodalAntServer(
        ip_address=config.ip_address,
        port=config.port,
        compression_type=config.compression_type,
        seed=config.seed,
        num_trials=config.num_trials,
        output_folder=config.output_folder,
        max_parallel_envs=config.max_parallel_envs,
        record_video=config.record_video,
    )
    logging.info(
        f"Multimodal Ant eval: {config.num_trials} trials "
        f"(seeds starting at {config.seed}), "
        f"waiting for client on tcp://{config.ip_address}:{config.port}"
    )

    try:
        while True:
            response = server.handle_client_request()
            if response.get(TransportKey.STATUS.value) == ServerStatus.FINISHED.value:
                break
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        server.shutdown()

    rollout_directory = server.environment.rollout_directory
    rollout_directory.mkdir(parents=True, exist_ok=True)
    log_filepath = str(rollout_directory / "log.txt")
    _log_results(server=server, config=config, log_filepath=log_filepath)
    logging.info(f"Log saved to: {log_filepath}")


def _log_results(
    server: MultimodalAntServer,
    config: EvalConfig,
    log_filepath: str,
) -> None:
    environment = server.environment
    if config.use_wandb:
        wandb.config.update(
            {
                "client_name": environment.client_name,
                "num_trials": config.num_trials,
                "seed": config.seed,
                "record_video": config.record_video,
            }
        )

    total_trials = sum(environment.number_of_resets)
    mean_goals = (
        sum(environment.environments_goals_achieved) / total_trials
        if total_trials > 0
        else 0.0
    )
    entropy = environment.first_goal_entropy()
    first_goal_counts = environment.first_goal_counts()
    behavior_order_counts = environment.behavior_order_counts()

    with open(log_filepath, "w") as log_file:
        log_file.write(
            f"Multimodal Ant evaluation - {config.num_trials} trials "
            f"(seeds {config.seed}..{config.seed + config.num_trials - 1})\n\n"
        )
        for i in range(environment.num_envs):
            episode_seed = environment._episode_seeds[i]
            goals_achieved = environment.environments_goals_achieved[i]
            behavior_order = environment.environments_behavior_order[i]
            log_file.write(
                f"Trial {i:3d} (seed={episode_seed}): "
                f"goals={goals_achieved}/4, "
                f"behavior_order={behavior_order}\n"
            )
            if config.use_wandb:
                wandb.log(
                    {
                        "episode": i,
                        f"goals/trial_{i}": goals_achieved,
                    }
                )

        log_file.write(f"\nTrials: {total_trials}\n")
        log_file.write(f"Mean goals (/4): {mean_goals:.4f}\n")
        log_file.write(f"First goal entropy: {entropy:.4f}\n")
        log_file.write(f"First goal counts: {dict(first_goal_counts)}\n")
        log_file.write(f"Behavior order counts: {dict(behavior_order_counts)}\n")

    if config.use_wandb:
        wandb.log(
            {
                "mean_goals": mean_goals,
                "first_goal_entropy": entropy,
                "num_episodes/total": total_trials,
            }
        )
        for label, count in first_goal_counts.items():
            wandb.log({f"first_goal_count/{label}": count})

    logging.info(
        f"Mean goals (/4): {mean_goals:.4f}, "
        f"first goal entropy: {entropy:.4f} "
        f"over {total_trials} trials"
    )


@draccus.wrap()
def eval_multimodal_ant(config: EvalConfig) -> None:
    run_evaluation(config=config)


if __name__ == "__main__":
    eval_multimodal_ant()
