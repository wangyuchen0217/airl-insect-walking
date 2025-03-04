import torch
import numpy as np
import gymnasium as gym

def collect_agent_trajectories(env, policy, steps_per_iter, device):
    """
    Run the current policy to collect a number of transitions.
    (Gymnasium’s reset() returns (obs, info) and step() returns (obs, reward, done, truncated, info)).
    """
    trajectories = []
    state, _ = env.reset()
    for _ in range(steps_per_iter):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, _ = policy.act(state_tensor)
        action_np = action.cpu().numpy()[0]
        next_state, reward, done, truncated, info = env.step(action_np)
        trajectories.append({
            'state': state,
            'action': action_np,
            'next_state': next_state,
            'log_prob': log_prob.cpu().item(),
            'done': done or truncated
        })
        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state
    return trajectories

def load_expert_data(model_name='Ant'):
    expert_states = torch.load(model_name+'_states.pt').numpy() # (62, 1000, 111)
    expert_actions = torch.load(model_name+'_actions.pt').numpy() # (62, 1000, 8)
    print("expert_states:", expert_states.shape, "expert_actions:", expert_actions.shape)
    # take only 1 trail
    expert_states = expert_states[1,:,:27] # (1000, 27)
    expert_actions = expert_actions[1] # (1000, 8)
    expert_next_states = np.roll(expert_states, -1, axis=0)
    expert_next_states[-1] = expert_states[-1]
    print("expert_states:", expert_states.shape, "expert_actions:", expert_actions.shape, "expert_next_states:", expert_next_states.shape)
    return expert_states, expert_actions, expert_next_states

if __name__ == "__main__":
    model_name = 'Ant'
    env = gym.make(model_name+"-v4", render_mode='human')
    env.reset_model()
    env.reset()

    # print the observation space and action space
    print("observation space:", env.observation_space)
    print("observation space shape:", env.observation_space.shape)
    print("action space:", env.action_space)
    print("action space shape:", env.action_space.shape)

    actions = torch.load(model_name+'_actions.pt').numpy()[1]

    for i in range(1000):
        action = actions[i]
        obs, reward, done, _, _=env.step(action)
        env.render()
        print("Step:", i, "Reward:", reward, "Done:", done)
    env.close()