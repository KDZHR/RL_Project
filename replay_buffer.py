from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self._storage = deque(maxlen=size)

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        '''
        Make sure, _storage will not exceed _maxsize.
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        '''
        data = (obs_t, action, reward, obs_tp1, done)
        self._storage.append(data)


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
        samples = [self._storage[i] for i in np.random.choice(len(self._storage), size=batch_size)]
            # randomly generate batch_size integers
            # to be used as indexes of samples


        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])
        next_states = np.array([sample[3] for sample in samples])
        is_done = np.array([sample[4] for sample in samples])
            # collect <s,a,r,s',done> for each index

        return states, actions, rewards, next_states, is_done
            # <states>, <actions>, <rewards>, <next_states>, <is_done>