import gymnasium as gym
import torch

states = torch.load("Ant_states.pt", weights_only=True)   # shape: (num_trajectories, traj_length, state_dim)
actions = torch.load("Ant_actions.pt", weights_only=True)   # shape: (num_trajectories, traj_length, action_dim)
    
# Convert to numpy arrays.
states_np = states.numpy()[1,:,:27]#[:,:,:27]
actions_np = actions.numpy()[1]
print(f"states: {states_np.shape}, actions: {actions_np.shape}")

# 创建环境并固定种子
env = gym.make('Ant-v4', render_mode='human')
reset_out = env.reset(seed=123)
if isinstance(reset_out, tuple):
    state, info = reset_out
else:
    state = reset_out

# 按照记录的动作顺序进行回放
done = False
i = 0
total_reward = 0
while not done:
    action = actions_np[i]
    obs, reward, done, _, _=env.step(action)
    env.render()
    i += 1
    total_reward += reward
    print("Step:", i, "Reward:", total_reward, "Done:", done)
env.close()