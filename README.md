# Multimodal Ant Inference Server for VersatIL

Standalone simulation server for evaluating VersatIL policies on the
multimodal Ant maze task. A VersatIL client process connects over ZMQ,
receives low-dimensional state observations, and sends back 8D joint torque
actions; the server drives the simulator, records rollouts, and computes
evaluation metrics.

An ant robot starts near the maze center with four goals at fixed corners.
An episode ends when all four goals are visited or after 1200 steps. The
per-trial metric is the number of goals reached; the order in which goals are
visited across trials measures the multimodality of the policy
(`first_goal_entropy`: 1.0 means the policy always commits to the same goal
first, 4.0 a uniform spread over all four).

The simulator dynamics and task logic are based on the multimodal ant
environment released with
[VQ-BeT](https://github.com/jayLEE0301/vq_bet_official/tree/main/envs/antenv)
([Behavior Generation with Latent Actions](https://arxiv.org/abs/2403.03181),
Lee et al., 2024), ported to gymnasium and the modern `mujoco` bindings. The
simulator code is vendored under `ant_sim/`.
The `versatil_inference/` package is only the environment-side wrapper. The
policy client lives in the VersatIL codebase and is run with
`python -m versatil.endpoints.deploy`.

The server uses raw simulator state/action values, because VersatIL
normalizes observations and unnormalizes actions inside the policy client
before sending actions to the environment server.

## Observations and Actions

Observation keys served to the client (from `versatil-constants`):

| Key | Dimension | Content |
|-----|-----------|---------|
| `ant_qpos` | 15 | Torso pose and joint positions. |
| `ant_qvel` | 14 | Torso and joint velocities. |
| `ant_goal_coords` | 8 | Zeroed, matching the unconditional training data. |
| `ant_achieved` | 4 | Per-goal achievement bits. |

Actions are 8D joint torques under the VersatIL structured action format.

## Layout

- `ant_sim/` - Ant maze simulator code and MJCF assets adapted from VQ-BeT's
  ant environment.
- `versatil_inference/` - ZMQ server, parallel episode manager, rollout
  recorder, and evaluation entry point.

## Install

Install Miniforge with `mamba` available if needed.

```bash
cd simulation_multimodal_ant
mamba env create -f environment.yml
mamba run -n multimodal_ant bash -lc \
  'UV_PROJECT_ENVIRONMENT=$MAMBA_ROOT_PREFIX/envs/multimodal_ant uv sync'
```

`uv sync` installs `versatil-constants>=0.2.1`, which provides the shared
multimodal Ant wire protocol constants used by both this server and VersatIL.

For headless rendering, set the MuJoCo backend before running:

```bash
export MUJOCO_GL=egl
```

## Run

Start the simulator server:

```bash
cd simulation_multimodal_ant
mamba activate multimodal_ant
python -m versatil_inference.run_evaluation \
  --num_trials 50 \
  --max_parallel_envs 10 \
  --port 5556 \
  --output_folder ./results/multimodal_ant \
  --record_video true \
  --use_wandb false
```

Then run the policy client from a VersatIL environment against that server:

```bash
python -m versatil.endpoints.deploy \
  checkpoint_path=/path/to/checkpoint_dir \
  client.model_server_address=127.0.0.1 \
  client.model_server_port=5556
```

For the full client documentation, see the
[VersatIL deployment tutorial](https://lorenzo-mazza.github.io/VersatIL/getting-started/inference/).

Per-trial results, rollout videos (`--record_video true`), torso trajectory
CSVs, and a `results.csv` summary (goals reached, behavior orders, first-goal
entropy) are written under the configured `output_folder`, or under the
client checkpoint's `rollouts/` directory when unset.

## Citation

If you use this benchmark, please cite the original environment:

```bibtex
@article{lee2024behavior,
    title={Behavior Generation with Latent Actions},
    author={Lee, Seungjae and Wang, Yibin and Etukuru, Haritheja and Kim, H Jin and Shafiullah, Nur Muhammad Mahi and Pinto, Lerrel},
    journal={arXiv preprint arXiv:2403.03181},
    year={2024}
}
```
