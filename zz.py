import torch
from common.buffer import SerializedBuffer
import gymnasium as gym

# read pth file
model = torch.load('weights/Hopper-v3.pth', weights_only=True)

expert_buffer = SerializedBuffer(
        path='/home/yuchen/airl_insect_walking/buffers/Hopper-v4/size1000000_std0.01_prand0.0.pth',
        device='cuda:0')
state = expert_buffer.states
action = expert_buffer.actions
reward = expert_buffer.rewards
dones = expert_buffer.dones
next_state = expert_buffer.next_states

print(f"state: {state.shape}")
print(f"action: {action.shape}")
print(f"reward: {reward.shape}")
print(f"dones: {dones.shape}")
print(f"next_state: {next_state.shape}")

# convert to csv and save
import pandas as pd
df = pd.DataFrame(state.cpu().numpy())
df.to_csv('state.csv', index=False)
df = pd.DataFrame(action.cpu().numpy())
df.to_csv('action.csv', index=False)
df = pd.DataFrame(reward.cpu().numpy())
df.to_csv('reward.csv', index=False)
df = pd.DataFrame(dones.cpu().numpy())
df.to_csv('dones.csv', index=False)
df = pd.DataFrame(next_state.cpu().numpy())
df.to_csv('next_state.csv', index=False)


# create gym environment
env = gym.make('Hopper-v4')
print(f"observation_space: {env.observation_space}")
print(f"action_space: {env.action_space}")