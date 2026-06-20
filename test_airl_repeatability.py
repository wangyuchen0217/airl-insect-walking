'''
This script evaluates a trained AIRL/PPO Actor model in CoppeliaSim
and computes mean speed ± SD and roll/pitch RMS ± SD across trials.
'''

import os
import sys
import torch
import numpy as np
from networks.actor import ActorNetworkPolicy
import logging
from common.base import LoggerWriter

# from common.normalized_env_red_mirror import CoppeliaSimEnv
# from common.normalized_env_66k import CoppeliaSimEnv
from common.normalized_env_66k_legloss import CoppeliaSimEnv
# from common.normalized_env_66k_RM_error import CoppeliaSimEnv


ENV_ID = "Medauroidea_66k_aug3c_legloss"
ALGO = "airl_logit_vx"
FILENAME = "20251103-1924"
PORT = 23000
CUDA = 0
NUM_EPISODES = 10
STEP_NUM = 840000
LOG = True

# State index settings
ROLL_IDX = 1
PITCH_IDX = 2

# Control frequency / simulation step
DT = 1.0 / 50.0   # change to 1/30 if evaluation is executed at 30 Hz


def compute_rms(x):
    x = np.asarray(x)
    return np.sqrt(np.mean(x ** 2))


def main():
    
    SAVE_PATH = os.path.join("repeatability_logs", ENV_ID, ALGO, FILENAME)
    LOG_PATH = os.path.join("logs", ENV_ID, ALGO, FILENAME)
    # Ensure save path exists before creating log files or using model paths
    os.makedirs(SAVE_PATH, exist_ok=True)

    if LOG:
        log_filename = os.path.join(SAVE_PATH, "evaluation_metrics.log")
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format="%(message)s",
            filemode="w"
        )
        sys.stdout = LoggerWriter(logging.info)

    device = torch.device(f"cuda:{CUDA}" if torch.cuda.is_available() and CUDA >= 0 else "cpu")

    env = CoppeliaSimEnv(port=PORT, OnTimeStep=True)

    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    actor = ActorNetworkPolicy(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_units=(64, 64),
        hidden_activation=torch.nn.Tanh()
    ).to(device)

    if STEP_NUM is None:
        actor_path = f"{LOG_PATH}/model/actor.pth"
        eval_dir = os.path.join(SAVE_PATH, "eval", "final")
    else:
        actor_path = f"{LOG_PATH}/model/step{STEP_NUM}/actor.pth"
        eval_dir = os.path.join(SAVE_PATH, "eval", f"step{STEP_NUM}")

    # Ensure model directory exists (useful if saving later or avoiding errors)
    model_dir = os.path.dirname(actor_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(actor_path):
        actor.load_state_dict(torch.load(actor_path, weights_only=True, map_location=device))
        print(f"Loaded actor model from {actor_path}")
    else:
        print(f"Actor model file not found: {actor_path}")
        return

    os.makedirs(eval_dir, exist_ok=True)

    actor.eval()

    episode_speeds = []
    episode_roll_rms = []
    episode_pitch_rms = []
    episode_returns = []
    episode_steps = []

    print("--- Evaluation ---")

    for ep in range(NUM_EPISODES):
        reset_out = env.reset()

        if isinstance(reset_out, tuple):
            state, _ = reset_out
        else:
            state = reset_out

        if isinstance(state, dict):
            state = state.get("observation", state)

        done = False
        ep_return = 0.0
        step = 0

        states = []
        actions = []
        body_x_positions = []
        roll_values = []
        pitch_values = []

        # Initial body x position
        x0 = env.sim.getObjectPosition(env.sim.getObject('/head'))[0]

        while not done:
            state_tensor = torch.tensor(
                np.array(state),
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():
                mean = actor(state_tensor)
                action = mean.cpu().numpy()[0]

            next_step = env.step(action)
            state, reward, terminated, truncated, info = next_step
            done = terminated or truncated

            ep_return += reward
            step += 1

            states.append(state)
            actions.append(action)

            roll_values.append(state[ROLL_IDX])
            pitch_values.append(state[PITCH_IDX])

            body_x = env.sim.getObjectPosition(env.sim.getObject('/head'))[0]

            body_x_positions.append(body_x)

        total_time = step * DT
        x_end = body_x_positions[-1]
        mean_speed = (x_end - x0) / total_time

        roll_rms = compute_rms(roll_values)
        pitch_rms = compute_rms(pitch_values)

        episode_speeds.append(mean_speed)
        episode_roll_rms.append(roll_rms)
        episode_pitch_rms.append(pitch_rms)
        episode_returns.append(ep_return)
        episode_steps.append(step)

        np.savetxt(
            os.path.join(eval_dir, f"episode_{ep+1}_states.csv"),
            np.array(states),
            delimiter=","
        )
        np.savetxt(
            os.path.join(eval_dir, f"episode_{ep+1}_actions.csv"),
            np.array(actions),
            delimiter=","
        )

        print(
            f"Episode {ep+1}: "
            f"Return={ep_return:.2f}, "
            f"Steps={step}, "
            f"MeanSpeed={mean_speed:.4f} m/s, "
            f"RollRMS={roll_rms:.4f}, "
            f"PitchRMS={pitch_rms:.4f}"
        )

    episode_speeds = np.array(episode_speeds)
    episode_roll_rms = np.array(episode_roll_rms)
    episode_pitch_rms = np.array(episode_pitch_rms)

    print("\n--- Summary across episodes ---")
    print(f"Number of trials: {NUM_EPISODES}")
    print(f"Mean speed: {episode_speeds.mean():.4f} ± {episode_speeds.std(ddof=1):.4f} m/s")
    print(f"Roll RMS: {episode_roll_rms.mean():.4f} ± {episode_roll_rms.std(ddof=1):.4f}")
    print(f"Pitch RMS: {episode_pitch_rms.mean():.4f} ± {episode_pitch_rms.std(ddof=1):.4f}")

    summary = np.column_stack([
        episode_speeds,
        episode_roll_rms,
        episode_pitch_rms,
        episode_returns,
        episode_steps
    ])

    np.savetxt(
        os.path.join(eval_dir, "evaluation_summary_each_trial.csv"),
        summary,
        delimiter=",",
        header="mean_speed_m_per_s,roll_rms,pitch_rms,return,steps",
        comments=""
    )

    summary_stats = np.array([
        [episode_speeds.mean(), episode_speeds.std(ddof=1)],
        [episode_roll_rms.mean(), episode_roll_rms.std(ddof=1)],
        [episode_pitch_rms.mean(), episode_pitch_rms.std(ddof=1)]
    ])

    np.savetxt(
        os.path.join(eval_dir, "evaluation_summary_mean_sd.csv"),
        summary_stats,
        delimiter=",",
        header="mean,sd\nmean_speed_m_per_s\nroll_rms\npitch_rms",
        comments=""
    )

    env.stop()


if __name__ == "__main__":
    main()