'''
This module implements various types of experience replay buffers for reinforcement learning.
It includes:
- SerializedBuffer: Loads a pre-saved buffer from disk and allows sampling.
- Buffer: A standard replay buffer that supports appending new experiences and saving to disk.
- RolloutBuffer: A buffer designed for storing rollouts, with functionality to sample from the most recent K iterations.
- ExpertBuffer: A buffer for storing expert demonstration data.
'''


import numpy as np
import torch
import os

class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path, weights_only=True)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)


class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)
        # print(f"RolloutBuffer: _n={self._n}, _p={self._p}, total_size={self.total_size}")

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )
    
    # -------------------- ADD: Mix samples from the previous k iterations -------------------- #
    def sample_last_k_iters(self, batch_size, k=20, include_current=True):
        """
        Uniformly sample batch_size transitions from the union of the most recent K rounds.
        K is counted as rounds, where 1 round = buffer_size transitions.
        When include_current=True, the most recently collected round is included.
        This function should be called at the end of each round (_p % buffer_size == 0).
        """
        assert self._p % self.buffer_size == 0, \
            "Call after a rollout is complete (p % buffer_size == 0)."

        # the number of completed iterations
        effective_iters = self._n // self.buffer_size
        if effective_iters == 0:
            raise RuntimeError("No completed iterations to sample from.")

        # check if the current round is included
        offset0 = 0 if include_current else 1
        available = max(0, effective_iters - offset0)
        if available == 0:
            raise RuntimeError("Not enough past iterations for the requested setting.")

        k = min(k, available)

        # Sample uniformly by round, then uniformly sample a position within that round.
        # n=0 represents the "most recent round" (or the current round if include_current=True ).
        # The starting point of the interval is (_p - buffer_size*(n+1)) % total_size
        # Within that interval, offset j∈[0, buffer_size)
        iter_choices = np.random.randint(low=0, high=k, size=batch_size)
        intra_choices = np.random.randint(low=0, high=self.buffer_size, size=batch_size)

        # Calculate the actual index 
        # (the whole segment is circular, but this formula automatically takes the modulus)
        idxes = np.empty(batch_size, dtype=np.int64)
        for t in range(batch_size):
            n = iter_choices[t] + offset0       # choose the n-th most recent round
            start = (self._p - self.buffer_size * (n + 1)) % self.total_size
            idxes[t] = (start + intra_choices[t]) % self.total_size

        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )
    
class ExpertBuffer:
    def __init__(self, expert_data, device='cpu'):
        """
        expert_data: dict with keys 'state' and 'action'
        """
        self.device = device
        self.states = torch.tensor(expert_data['state'], dtype=torch.float32, device=device)
        self.actions = torch.tensor(expert_data['action'], dtype=torch.float32, device=device)
        self.size = self.states.shape[0]

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.states[idxs], self.actions[idxs]