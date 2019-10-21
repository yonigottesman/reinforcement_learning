import os
import random
from collections import deque

import numpy as np
import torch.nn as nn
import torch
import gym
import torch.nn.functional as F

from common.memory import Experience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v1')

if not os.path.exists('models'):
    os.makedirs('models')


MODEL_PATH = 'models/cartpole.pt'


class Policy(nn.Module):
    def __init__(self, state_shape, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_shape, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, n_actions)

    def forward(self, state):
        a2 = F.relu(self.fc1(state))
        a3 = F.relu(self.fc2(a2))
        a4 = self.fc3(a3)
        return a4


def discount_rewards(rewards, dones):
    n = len(rewards)
    discounted_rewards = torch.zeros(n)
    running_add = 0
    for t in reversed(range(0, n)):
        if dones[t]:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add

    return discounted_rewards


# training hyperparams
lr = 0.01
gamma = 0.99
episodes_batch_size = 32  # number of episodes until training

# (state, reward, done)
def learn(policy, states, actions, rewards, dones, optimizer):
    discounted_rewards = discount_rewards(rewards, dones)
    log_probs = (F.log_softmax(policy(torch.tensor(states).float()), dim=1)
                 .gather(1, torch.tensor(actions).unsqueeze(1))
                 .squeeze())
    # log_probs2 = F.log_softmax(policy(torch.tensor(states).float()), dim=1)[range(len(experiences.action)), torch.tensor(experiences.action)]
    loss = -torch.mean(log_probs*discounted_rewards)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    return loss


def predict_action(policy, state):

    with torch.no_grad():
        action_prob = F.softmax(policy(torch.tensor(state).float()), dim=0)
        probs = action_prob.data.cpu().numpy()  # magic that fixes precision
    action = np.random.choice(env.action_space.n, p=probs)
    return action


def train():
    policy = Policy(state_shape=env.observation_space.shape[0],
                    n_actions=env.action_space.n).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    rewards_list = deque([], maxlen=100)

    states = []
    actions = []
    rewards = []
    dones = []

    for episode in range(5000):
        episode_rewards = 0
        state = env.reset()
        done = False
        while not done:
            action = predict_action(policy, state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            episode_rewards += reward
            state = next_state

            if done:
                rewards_list.append(episode_rewards)
                moving_average = np.mean(rewards_list)
                if episode % 50 == 0:
                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(episode_rewards),
                          #'Training Loss {}'.format(loss),
                          'Moving average {}'.format(moving_average))

                if episode % episodes_batch_size == 0:
                    loss = learn(policy, states, actions, rewards, dones, optimizer)
                    states.clear()
                    actions.clear()
                    rewards.clear()
                    dones.clear()



        if episode % 100 == 0:
            torch.save(policy.state_dict(), MODEL_PATH)


def play():
    dqn = Policy(state_shape=env.observation_space.shape[0],
                 n_actions=env.action_space.n)
    dqn.load_state_dict(torch.load(MODEL_PATH))

    while True:
        state = env.reset()
        done = False
        episode_score = 0
        while not done:
            with torch.no_grad():
                qs = dqn(torch.from_numpy(state).float())
            action = torch.argmax(qs).item()
            next_state, reward, done, _ = env.step(action)
            if done:
                for i in range(100):
                    env.render()
                    env.step(random.randint(0, env.action_space.n - 1))
            episode_score += reward
            env.render()
            state = next_state
        print('episode score {}'.format(episode_score))



def main():
    #train()
    play()


if __name__ == '__main__':
    main()
