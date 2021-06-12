#code from openai
#https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import numpy as np
import random, torch, operator
from copy import deepcopy


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient `reduce`
               operation which reduces `operation` over
               a contiguous subsequence of items in the
               array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must for a mathematical group together with the set of
            possible values for array elements.
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class ReplayBuffer(object):
    def __init__(self, n_agents, state_dim, size, device="cpu"):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._maxsize = size
        
        self.n_agents = n_agents
        self.device = device
        
        self.state_dicts = [None for _ in range(self._maxsize)]
        self.next_state_dicts = [None for _ in range(self._maxsize)]
        self.global_states = torch.empty((self._maxsize, state_dim), dtype=torch.float, 
                                   device=self.device)
        self.agent_states = torch.empty((n_agents, self._maxsize, state_dim), 
                                        dtype=torch.float, device=self.device)
        self.actions = torch.empty((n_agents, self._maxsize), dtype=torch.float, 
                                   device=self.device)
        self.next_global_states = torch.empty((self._maxsize, state_dim), dtype=torch.float, 
                                        device=self.device)
        self.next_agent_states = torch.empty((n_agents, self._maxsize, state_dim), 
                                             dtype=torch.float, device=self.device)
        self.rewards = torch.empty((n_agents, self._maxsize), dtype=torch.float, 
                                   device=self.device)
        self.dones = torch.empty((self._maxsize), dtype=torch.float, device=self.device)
        
        
        self.pos = 0
        self.cur_size = 0

    def __len__(self):
        return self.cur_size

    def push(self, transition):
        if self.cur_size < self._maxsize:
            self.cur_size += 1
            
        (state_dict, next_state_dict, gstate, agent_states, actions, 
         next_gstates, next_agent_states, rewards, dones) = transition
        
        self.state_dicts[self.pos] = deepcopy(state_dict)
        self.next_state_dicts[self.pos] = deepcopy(next_state_dict)
        self.global_states[self.pos] = torch.tensor(gstate, dtype=torch.float, device=self.device)
        self.next_global_states[self.pos] = torch.tensor(next_gstates, 
                                                         dtype=torch.float, 
                                                         device=self.device)
        self.dones[self.pos] = torch.tensor(dones, dtype=torch.float, device=self.device)
        
        for idx in range(self.n_agents):
            self.agent_states[idx, self.pos] = torch.tensor(agent_states[idx], 
                                                            dtype=torch.float, 
                                                            device=self.device)
            self.next_agent_states[idx, self.pos] = torch.tensor(next_agent_states[idx], 
                                                                 dtype=torch.float, 
                                                                 device=self.device)
            self.rewards[idx, self.pos] = torch.tensor(rewards[idx], dtype=torch.float, device=self.device)
            self.actions[idx, self.pos] = torch.tensor(actions[idx], dtype=torch.float, device=self.device)
        self.pos = (self.pos + 1) % self._maxsize
        
        # if self._next_idx >= len(self._storage):
        #     self._storage.append(transition)
        # else:
        #     self._storage[self._next_idx] = transition
        # self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        # sd = []
        # nsd = []
        # global_states = []
        # agent_states = []
        # actions = []
        # next_global_states = []
        # next_agent_states = [] 
        # rewards = [] 
        # dones = []
        # for i in idxes:
        #     (state_dict, next_state_dict, gstates, a_s, action, next_gstates, 
        #      na_s, reward, done) = self._storage[i]
        #     sd.append(state_dict)
        #     nsd.append(next_state_dict)
        #     global_states.append(np.array(gstates, copy=False))
        #     agent_states.append(np.array(a_s, copy=False))
        #     actions.append(np.array(action, copy=False))
        #     next_global_states.append(np.array(next_gstates, copy=False))
        #     next_agent_states.append(np.array(na_s, copy=False))
        #     rewards.append(reward)
        #     dones.append(done)
        # return (sd, nsd, np.array(global_states), np.array(agent_states), 
        #         np.array(actions), np.array(next_global_states), 
        #         np.array(next_agent_states), np.array(rewards), np.array(dones))
        state_dicts = []
        next_state_dicts = []
        for idx in idxes:
            state_dicts.append(self.state_dicts[idx])
            next_state_dicts.append(self.next_state_dicts[idx])
        return (state_dicts, next_state_dicts,
                self.global_states[idxes], 
                self.agent_states[:, idxes],
                self.actions[:, idxes], 
                self.next_global_states[idxes], 
                self.next_agent_states[:, idxes], 
                self.rewards[:, idxes], 
                self.dones[idxes])

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        # idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        # return self._encode_sample(idxes)
        assert self.__len__() >= batch_size
        ids = np.random.choice(self.__len__(), batch_size, replace=False)
        return self._encode_sample(ids)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, n_agents, state_dim, size, alpha, device="cpu"):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(n_agents, state_dim, size, device=device)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def push(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self.pos
        super(PrioritizedReplayBuffer, self).push(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, self.__len__() - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.__len__()) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.__len__()) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample, (weights, idxes)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.__len__()
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)