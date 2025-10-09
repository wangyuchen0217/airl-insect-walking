import os
import sys
import torch
import numpy as np
from networks.actor import ActorNetworkPolicy 
from networks.discrim import AIRLDiscrim
import logging
from common.base import LoggerWriter
from common.normalized_env_60k import CoppeliaSimEnv
import matplotlib.pyplot as plt
import math
import pandas as pd 


# ----------------------------- helpers: ranges & grid -----------------------------------
def _get_base_state(env, mode="center", median_state=None):
    """
    Get a base state vector used to fix the other (unused) dimensions during slicing.
    mode:
      - "center": midpoint of observation_space.low/high
      - "reset":  one observation from env.reset()
      - "given":  use provided 'median_state' (e.g., dataset median)
    """
    # low = np.asarray(env.observation_space_low, dtype=np.float32)
    # high = np.asarray(env.observation_space_high, dtype=np.float32)
    low = np.zeros(env.observation_space.shape, dtype=np.float32)
    high = np.ones(env.observation_space.shape, dtype=np.float32)
    if mode == "center":
        return (low + high) / 2.0
    elif mode == "reset":
        try:
            s, *_ = env.reset()
        except:
            s = env.reset()
        return np.asarray(s, dtype=np.float32)
    elif mode == "given":
        assert median_state is not None, "median_state must be provided for mode='given'."
        return np.asarray(median_state, dtype=np.float32)
    else:
        raise ValueError(f"Unknown base state mode: {mode}")

def _get_axis_range(env, dim, mode="space", data_percentiles=None, p_low=1, p_high=99):
    """
    Decide the plotting range for one state dimension.
    mode:
      - "space": use observation_space.low/high
      - "data":  use percentiles from provided data samples (array of that dim)
    """
    # low = float(env.observation_space_low[dim])
    # high = float(env.observation_space_high[dim])
    low = np.zeros(env.observation_space.shape, dtype=np.float32)[dim]
    high = np.ones(env.observation_space.shape, dtype=np.float32)[dim]
    if mode == "space" or data_percentiles is None:
        return low, high
    v = np.asarray(data_percentiles, dtype=np.float32).reshape(-1)
    # 'data_percentiles' should be the samples of that dimension
    lo = float(np.percentile(v, p_low))
    hi = float(np.percentile(v, p_high))
    # fall back if pathological
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return low, high
    return lo, hi

def _build_slice_grid(env, base_state, dim_x, dim_y, grid_size=120,
                      range_mode="space", data_x=None, data_y=None,
                      p_low=1, p_high=99):
    """
    Build a grid slice over two chosen state dimensions, keeping others fixed at base_state.
    range_mode:
      - "space": use observation_space.low/high
      - "data":  use percentiles from data_x/data_y
    Returns: (grid_states [N, S], xs, ys, (x_low,x_high,y_low,y_high))
    """
    x_low, x_high = _get_axis_range(env, dim_x, range_mode, data_x, p_low, p_high)
    y_low, y_high = _get_axis_range(env, dim_y, range_mode, data_y, p_low, p_high)

    xs = np.linspace(x_low, x_high, grid_size, dtype=np.float32)
    ys = np.linspace(y_low, y_high, grid_size, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)  # [gy, gx]
    grid_states = np.repeat(base_state[None, :], grid_size * grid_size, axis=0)
    grid_states[:, dim_x] = XX.reshape(-1)
    grid_states[:, dim_y] = YY.reshape(-1)
    return grid_states, xs, ys, (x_low, x_high, y_low, y_high)

# ----------------------------- helpers: actor interface ----------------------------------
@torch.no_grad()
def _actor_forward_mu_logstd(actor, s_tensor):
    """
    Try to extract (mu, log_std) from various actor APIs.
    1) actor.evaluate_log_pi exists -> we won't use it here; just return forward outputs.
    2) forward returns (mu, log_std) or (mu, std)
    3) forward returns mu only -> log_std=None
    """
    out = actor(s_tensor)
    if isinstance(out, (tuple, list)):
        mu = out[0]
        if len(out) >= 2:
            log_std = out[1]
            # Some actors might return std instead of log_std; try to detect
            if torch.mean(torch.abs(log_std)) > 50:  # heuristic: huge magnitude -> maybe not log_std
                log_std = torch.log(torch.clamp(log_std, min=1e-6))
        else:
            log_std = None
    else:
        mu = out
        log_std = None
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
    return mu, log_std

@torch.no_grad()
def _compute_log_pi(actor, states, actions):
    """
    Compute log pi(a|s).
    Priority:
      1) If actor has 'evaluate_log_pi', use it.
      2) Else, assume diagonal Gaussian with outputs (mu, log_std).
    """
    # Case 1: dedicated API
    if hasattr(actor, "evaluate_log_pi"):
        return actor.evaluate_log_pi(states, actions)

    # Case 2: diagonal Gaussian
    mu, log_std = _actor_forward_mu_logstd(actor, states)
    if log_std is None:
        # Deterministic policy: approximate with tiny std
        log_std = torch.full_like(mu, -5.0)  # std ~= exp(-5) ~ 0.0067

    var = torch.exp(2.0 * log_std)  # since log_std is log(std), var = exp(2*log_std)
    # Diagonal Normal log-prob: sum over action dims
    log_prob = -0.5 * (((actions - mu) ** 2) / (var + 1e-8) + 2.0 * log_std + math.log(2.0 * math.pi))
    return torch.sum(log_prob, dim=-1, keepdim=True)

@torch.no_grad()
def _sample_actions(actor, states, n_samples=1):
    """
    Sample actions from actor(s): returns [N, A] if n_samples=1 else [n_samples, N, A].
    """
    mu, log_std = _actor_forward_mu_logstd(actor, states)
    if log_std is None:
        # Deterministic fallback
        if n_samples == 1:
            return mu
        else:
            return mu.unsqueeze(0).repeat(n_samples, 1, 1)

    std = torch.exp(log_std)
    if n_samples == 1:
        eps = torch.randn_like(mu)
        return mu + eps * std
    else:
        Ns, A = mu.shape
        eps = torch.randn(n_samples, Ns, A, device=mu.device)
        return mu.unsqueeze(0) + eps * std.unsqueeze(0)

# ----------------------------- helpers: next-state choices --------------------------------
@torch.no_grad()
def _make_next_states(method, env, actor, states, device, gamma=0.99, n_samples=1):
    """
    Create next_states based on a chosen approximation method:
      - "same":   s' = s (rough trend; cheap)
      - "rollout": one-step in env using actor mean action (requires env stepping)
      - "none":   return None (caller must handle)
    Returns: next_states tensor [N, S] or list[ tensor ] if n_samples > 1 (for rollout sampling)
    """
    if method == "same":
        return states.clone()
    elif method == "none":
        return None
    elif method == "rollout":
        # WARNING: this will mutate the real env if used directly.
        # In practice, you should use a cloned/dummy env or reset before/after.
        # Here we provide a minimal version: one-step from current env state is non-trivial
        # unless the env exposes a setter for its full internal state. So we fallback to 'same'.
        print("[WARN] 'rollout' next_state requires stepping from each grid state. "
              "Unless you can set sim state deterministically, use 'same' or a learned dynamics model.")
        return states.clone()
    else:
        raise ValueError(f"Unknown next_state method: {method}")
    
# -------------------------------- plotting utility ----------------------------------------
def _save_heatmap(Z, xs, ys, extent, xlabel, ylabel, title, save_path):
    """
    Save a 2D heatmap where the colorbar range is simply [min(Z), max(Z)].
    No percentile clipping, no manual centering.
    """
    Znp = np.asarray(Z, dtype=np.float32)

    vmin = float(np.nanmin(Znp))  # use data-driven range
    vmax = float(np.nanmax(Znp))

    plt.figure(figsize=(6.0, 5.2))
    im = plt.imshow(
        Znp, origin="lower",
        extent=[extent[0], extent[1], extent[2], extent[3]],
        aspect="auto",
        vmin=vmin, vmax=vmax   # <- colorbar bounds from actual values
    )
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label("value")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"[Saved] {save_path}")

# -------------------------------- main visualizers ----------------------------------------
@torch.no_grad()
def visualize_g_heatmap(env, disc, device, dim_x, dim_y,
                        grid_size=120, base_mode="center",
                        range_mode="space", data_x=None, data_y=None,
                        save_path="heatmap_g.png"):
    """
    Plot g(s) over a 2D slice of the state space.
    """
    base = _get_base_state(env, base_mode)
    grid_states, xs, ys, extent = _build_slice_grid(
        env, base, dim_x, dim_y, grid_size, range_mode, data_x, data_y
    )
    s = torch.from_numpy(grid_states).to(device)
    # g(s): [N,1] or [N]
    g = disc.g(s)
    g = g.view(grid_size, grid_size).detach().cpu().numpy()
    print("g:", g)
    print("g shape:", g.shape)
    _save_heatmap(
        g, xs, ys, [extent[0], extent[1], extent[2], extent[3]],
        xlabel=f"s[{dim_x}]", ylabel=f"s[{dim_y}]",
        title="g(s) ~ recovered reward", save_path=save_path
    )

@torch.no_grad()
def visualize_f_heatmap(env, disc, device, dim_x, dim_y, gamma,
                        grid_size=120, base_mode="center",
                        range_mode="space", data_x=None, data_y=None,
                        next_state_method="same",
                        save_path="heatmap_f.png"):
    """
    Plot f(s,s') = g(s) + gamma*h(s') - h(s) over a 2D slice.
    """
    base = _get_base_state(env, base_mode)
    grid_states, xs, ys, extent = _build_slice_grid(
        env, base, dim_x, dim_y, grid_size, range_mode, data_x, data_y
    )
    s = torch.from_numpy(grid_states).to(device)
    s_next = _make_next_states(next_state_method, env, None, s, device, gamma)
    if s_next is None:
        s_next = s

    g = disc.g(s)
    h_s = disc.h(s)
    h_sp = disc.h(s_next)
    f = g + gamma * h_sp - h_s
    f = f.view(grid_size, grid_size).detach().cpu().numpy()
    _save_heatmap(
        f, xs, ys, [extent[0], extent[1], extent[2], extent[3]],
        xlabel=f"s[{dim_x}]", ylabel=f"s[{dim_y}]",
        title="f(s,s') = g(s)+γh(s')-h(s)", save_path=save_path
    )

@torch.no_grad()
def visualize_logit_heatmap(env, disc, actor, device, dim_x, dim_y, gamma,
                            grid_size=120, base_mode="center",
                            range_mode="space", data_x=None, data_y=None,
                            next_state_method="same",
                            action_mode="mean",   # "mean" or "sample"
                            samples_per_state=5,
                            save_path="heatmap_logit.png"):
    """
    Plot logit(s,a,s') = g(s) + γh(s') - h(s) - log pi(a|s).
    """
    base = _get_base_state(env, base_mode)
    grid_states, xs, ys, extent = _build_slice_grid(
        env, base, dim_x, dim_y, grid_size, range_mode, data_x, data_y
    )
    s = torch.from_numpy(grid_states).to(device)
    s_next = _make_next_states(next_state_method, env, actor, s, device, gamma)
    if s_next is None:
        s_next = s

    # g/h terms
    g = disc.g(s)            # [N,1]
    h_s = disc.h(s)
    h_sp = disc.h(s_next)
    f_val = g + gamma * h_sp - h_s  # [N,1]

    # action term
    if action_mode == "mean":
        mu, _ = _actor_forward_mu_logstd(actor, s)  # [N,A]
        actions = mu
        log_pi = _compute_log_pi(actor, s, actions)  # [N,1]
        logit = f_val - log_pi
    else:
        # Monte-Carlo over actions to approximate E_a[logit]
        acc = torch.zeros_like(f_val)
        for _ in range(samples_per_state):
            a = _sample_actions(actor, s, n_samples=1)  # [N,A]
            log_pi = _compute_log_pi(actor, s, a)       # [N,1]
            acc += (f_val - log_pi)
        logit = acc / float(samples_per_state)

    logit = disc(s, 0, log_pi, s_next)
    print("logit:", logit)
    print("logit shape:", logit.shape)

    Z = logit.view(grid_size, grid_size).detach().cpu().numpy()
    _save_heatmap(
        Z, xs, ys, [extent[0], extent[1], extent[2], extent[3]],
        xlabel=f"s[{dim_x}]", ylabel=f"s[{dim_y}]",
        title="logit(s,a,s')", save_path=save_path
    )
# ======================= end of Visualizers =======================












def load_expert_data(expert_file):

    # load the expert data (CoppeliaSim)
    data = pd.read_csv(expert_file, header=None)
    states_np = data[[
                                    'body_x', 'body_y',
                                    'body_z', 
                                    'body_roll', 'body_pitch', 'body_yaw', 
                                    'motor_pos_FL_TC', 'motor_pos_FL_CF', 'motor_pos_FL_FT', 
                                    'motor_pos_ML_TC', 'motor_pos_ML_CF', 'motor_pos_ML_FT',
                                    'motor_pos_HL_TC', 'motor_pos_HL_CF', 'motor_pos_HL_FT',
                                    'motor_pos_FR_TC', 'motor_pos_FR_CF', 'motor_pos_FR_FT',
                                    'motor_pos_MR_TC', 'motor_pos_MR_CF', 'motor_pos_MR_FT',
                                    'motor_pos_HR_TC', 'motor_pos_HR_CF', 'motor_pos_HR_FT',
                                    # 'qvel_body_x', 'qvel_body_y', 'qvel_body_z', 'qvel_body_roll', 'qvel_body_pitch', 'qvel_body_yaw',
                                    # 'qvel_FL_TC', 'qvel_FL_CF', 'qvel_FL_FT',
                                    # 'qvel_ML_TC', 'qvel_ML_CF', 'qvel_ML_FT',
                                    # 'qvel_HL_TC', 'qvel_HL_CF', 'qvel_HL_FT',
                                    # 'qvel_FR_TC', 'qvel_FR_CF', 'qvel_FR_FT',
                                    # 'qvel_MR_TC', 'qvel_MR_CF', 'qvel_MR_FT',
                                    # 'qvel_HR_TC', 'qvel_HR_CF', 'qvel_HR_FT',
                                    'force_FL', 'force_ML', 'force_HL', 'force_FR', 'force_MR', 'force_HR',
                                    'FL_foot_traj_z', 'ML_foot_traj_z', 'HL_foot_traj_z', 'FR_foot_traj_z', 'MR_foot_traj_z', 'HR_foot_traj_z',
                                    # 'contact_FL', 'contact_ML', 'contact_HL', 'contact_FR', 'contact_MR', 'contact_HR'
                                ]].values
    actions_np = data[[
                                    'motor_cmd_FL_TC', 'motor_cmd_FL_CF', 'motor_cmd_FL_FT',
                                    'motor_cmd_ML_TC', 'motor_cmd_ML_CF', 'motor_cmd_ML_FT',
                                    'motor_cmd_HL_TC', 'motor_cmd_HL_CF', 'motor_cmd_HL_FT',
                                    'motor_cmd_FR_TC', 'motor_cmd_FR_CF', 'motor_cmd_FR_FT',
                                    'motor_cmd_MR_TC', 'motor_cmd_MR_CF', 'motor_cmd_MR_FT',
                                    'motor_cmd_HR_TC', 'motor_cmd_HR_CF', 'motor_cmd_HR_FT'
                                ]].values

    states = []
    actions = []
    next_states = []
    for i in range(len(states_np) - 1):
        states.append(states_np[i])
        actions.append(actions_np[i])
        next_states.append(states_np[i + 1])

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    rewards = np.zeros(len(states), dtype=np.float32)  # Assuming zero rewards for expert data (not affect AIRL training)
    dones = np.zeros(len(states), dtype=np.float32)  # Assuming no terminal states for expert data

    expert_data = {
        'state': states,
        'action': actions,
        'reward': rewards,
        'done': dones,
        'next_state': next_states
    }
    # print(f"Expert data states: {expert_data['state'].shape}, actions: {expert_data['action'].shape}")
    # print(expert_data)

    return expert_data


# ======== Parameters (modify these as needed) =========
ENV_ID = "Medauroidea_60000_offset"
ALGO = "airl_logit"
FILENAME = "20250819-1650" 
PORT = 23000 # CoppeliaSim port: default is 23000
CUDA = 0
NUM_EPISODES = 5
STEP_NUM =1250000  # Choose a certain step number of the saved model or None 
# =================================================


SAVE_PATH = os.path.join("logs", ENV_ID, ALGO, FILENAME)

# Set the device and env
device = torch.device(f"cuda:{CUDA}" if torch.cuda.is_available() and CUDA >= 0 else "cpu")
OnTimeStep = True
env = CoppeliaSimEnv(port=PORT, OnTimeStep=OnTimeStep, simulation = False)

# Get state and action shapes from the environment
state_shape = env.observation_space.shape    # e.g., (28,)
action_shape = env.action_space.shape          # e.g., (18,)
print(f"State shape: {state_shape}, Action shape: {action_shape}")

# Instantiate the Actor network with the same architecture as used during training
actor = ActorNetworkPolicy(
    state_shape=state_shape,
    action_shape=action_shape,
    hidden_units=(64, 64),
    hidden_activation=torch.nn.Tanh()
).to(device)

disc = AIRLDiscrim(
            state_shape, 
            gamma=0.995,
            hidden_units_r=(100, 100),
            hidden_units_v=(100, 100), 
            hidden_activation_r=torch.nn.ReLU(inplace=True),
            hidden_activation_v=torch.nn.ReLU(inplace=True)
).to(device)

# Load the saved actor model parameters from a .pth file
if STEP_NUM is None:
    actor_path = f"{SAVE_PATH}/model/actor.pth"
    disc_path = f"{SAVE_PATH}/model/discriminator.pth"
else:
    actor_path = f"{SAVE_PATH}/model/step{STEP_NUM}/actor.pth"
    disc_path = f"{SAVE_PATH}/model/step{STEP_NUM}/discriminator.pth"
if os.path.exists(actor_path):
    actor.load_state_dict(torch.load(actor_path, weights_only=True, map_location=device))
    print(f"Loaded actor model from {actor_path}")
    disc.load_state_dict(torch.load(disc_path, weights_only=True, map_location=device))
    print(f"Loaded discriminator model from {disc_path}")

# ---------- Visualization calls (choose state dims you care about) ----------
dim_x, dim_y = 2, 21       # pick two interpretable state indices
grid = 120                 # 80~160 is common
gamma = 0.995              # use your training gamma

# 1) g(s): recovered reward (policy-independent)
visualize_g_heatmap(env, disc,
                    device=device, dim_x=dim_x, dim_y=dim_y,
                    grid_size=grid, base_mode="center",
                    range_mode="space", save_path="heatmap_g.png")

# 2) f(s,s'): include shaping terms h(s), h(s')
visualize_f_heatmap(env, disc,
                    device=device, dim_x=dim_x, dim_y=dim_y, gamma=gamma,
                    grid_size=grid, base_mode="center",
                    range_mode="space", next_state_method="same",
                    save_path="heatmap_f.png")

# 3) logit(s,a,s'): “expert-likeness” under the current policy
visualize_logit_heatmap(env, disc,
                    actor=actor, device=device, dim_x=dim_x, dim_y=dim_y, gamma=gamma,
                    grid_size=grid, base_mode="center",
                    range_mode="space", next_state_method="same",
                    action_mode="mean", samples_per_state=5,
                    save_path="heatmap_logit.png")    