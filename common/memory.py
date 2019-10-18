import random
from collections import namedtuple

import numpy as np

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        batch = Experience(*zip(*batch))
        return batch

    def __len__(self):
        return len(self.memory)


def fill_memory(memory, env, n):
    possible_actions = np.identity(env.action_space.n, dtype=int)
    state = env.reset()
    for i in range(n):

        # Get the next_state, the rewards, done by taking a random action
        action = random.randint(1, len(possible_actions)) - 1
        next_state, reward, done, _ = env.step(action)

        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)

            # Add experience to memory
            memory.push(state, action, reward, next_state, done)

            # Start a new episode
            state = env.reset()

        else:
            # Add experience to memory
            memory.push(state, action, reward, next_state, done)

            # Our new state is now the next_state
            state = next_state
