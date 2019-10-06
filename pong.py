import os
import random
import time

import numpy as np
import torch.nn as nn
import torch
import gym
import torch.nn.functional as F

from Explorer import Explorer
from PaperExplorer import PaperExplorer
from StackedFrames import StackedFrames
from memory import ReplayMemory, fill_memory
from skimage import transform
from datetime import datetime
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('Pong-v0')

if not os.path.exists('models'):
    os.makedirs('models')


MODEL_PATH = 'models/pong.pt'
PROCESSED_FRAME_SIZE = [84, 84]


class DQN(nn.Module):

    # state_shape only hight and width
    def __init__(self, state_shape, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        flatten_size = DQN.conv_size_out(np.array(state_shape), 8, 4)
        flatten_size = DQN.conv_size_out(flatten_size, 4, 2)
        flatten_size = flatten_size.prod()*32  # n output channels
        self.fc1 = nn.Linear(flatten_size, 256)
        self.fc2 = nn.Linear(256, n_actions)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv1.weight)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    @staticmethod
    def conv_size_out(input_size, kernel_size, stride, padding=0):
        return (input_size + 2*padding - kernel_size)//stride + 1

    @staticmethod
    def flatten(x):
        return x.view(x.size(0), -1)

    def forward(self, state):

        normalized = state.float() / 255
        a2 = F.relu(self.conv1(normalized))
        a3 = F.relu(self.conv2(a2))
        a4 = F.relu(self.fc1(DQN.flatten(a3)))
        output = self.fc2(a4)

        return output


def process_frame(frame):
    gray = np.mean(frame, axis=2).astype(np.uint8)
    cropped_frame = gray[25:-12, 4:-4]
    resized = transform.resize(cropped_frame,
                               PROCESSED_FRAME_SIZE,
                               preserve_range=True).astype(np.uint8)
    return resized


# training hyperparams
memory_size = 50000
prefill_memory = 10000
batch_size = 32
lr = 1e-4
gamma = 0.99  # Discounting rate


def learn(dqn, memory, criterion, optimizer):
    a = datetime.now()

    batch = memory.sample(batch_size)

    rewards = torch.tensor(batch.reward).to(device)
    states = torch.cat(batch.state).to(device)
    actions = torch.tensor(batch.action).view(-1, 1).to(device)
    next_states = torch.cat(batch.next_state).to(device)
    b = datetime.now()
    delta = b - a
    dones = torch.tensor(batch.done).to(device)

    with torch.no_grad():
        next_state_qs = dqn(next_states[dones == False]).to(device)

    q_expected = rewards
    q_expected[dones == False] += \
        gamma * torch.max(next_state_qs, dim=1).values

    # get q values only of played moves
    q_predicted = dqn(states).gather(1, actions).squeeze().to(device)
    loss = criterion(q_predicted, q_expected)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    #print(delta.total_seconds()*1000000) # total milisec
    return loss


def predict_action(dqn, explorer, state, n_actions):
    if explorer.explore():
        # exploration
        action = random.randint(0, n_actions - 1)
    else:
        # exploitation
        with torch.no_grad():
            qs = dqn(state.to(device))
        action = torch.argmax(qs).item()

    return action, explorer.explore_prob()


def fill_memory(memory):
    frame_stack = StackedFrames(4, PROCESSED_FRAME_SIZE)

    state = env.reset()
    state = frame_stack.push_get(process_frame(state), True)

    # same number as paper
    for i in range(prefill_memory):
        action = random.randint(0, env.action_space.n - 1)
        next_state, reward, done, _ = env.step(action)
        next_state = frame_stack.push_get(process_frame(next_state))
        memory.push(state, action, reward, next_state, done)
        if done:
            state = env.reset()
            state = frame_stack.push_get(process_frame(state), True)
        else:
            state = next_state


def train():
    memory = ReplayMemory(memory_size)
    fill_memory(memory)
    print('finished filling memory')
    explorer = Explorer(1, 0.02, 1e-05)
    dqn = DQN(state_shape=PROCESSED_FRAME_SIZE, n_actions=env.action_space.n)\
        .to(device)

    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(dqn.parameters(), lr=lr)

    frame_stack = StackedFrames(4, PROCESSED_FRAME_SIZE)
    rewards_list = []
    total_steps = 0
    for episode in range(5000):
        episode_rewards = 0
        state = env.reset()
        state = frame_stack.push_get(process_frame(state), True)
        done = False
        while not done:
            a = datetime.now()
            total_steps += 1
            action, explore_probability = predict_action(dqn,
                                                         explorer,
                                                         state,
                                                         env.action_space.n)

            next_state, reward, done, _ = env.step(action)
            next_state = frame_stack.push_get(process_frame(next_state))

            env.render()
            episode_rewards += reward
            memory.push(state, action, reward, next_state, done)
            state = next_state

            loss = learn(dqn, memory, criterion, optimizer)
            b = datetime.now()
            delta = b - a
            #print(delta.total_seconds()*1000000) # total milisec
            if done:
                rewards_list.append(episode_rewards)

                print('Episode: {}'.format(episode),
                      'Total reward: {}'.format(episode_rewards),
                      'Explore P: {:.4f}'.format(explore_probability),
                      'Training Loss {}'.format(loss),
                      'total steps {}'.format(total_steps))

        if episode % 10 == 0:
            torch.save(dqn.state_dict(), MODEL_PATH)


def play():
    dqn = DQN(state_shape=PROCESSED_FRAME_SIZE,
              n_actions=env.action_space.n)

    dqn.load_state_dict(torch.load('models/pong.pt',  map_location=torch.device('cpu')))
    frame_stack = StackedFrames(4, PROCESSED_FRAME_SIZE)
    explorer = Explorer(0, 0, 0)
    for episode in range(5000):
        state = env.reset()
        state = frame_stack.push_get(process_frame(state), True)
        done = False
        episode_score = 0
        while not done:
            time.sleep(0.02)
            action, explore_probability = predict_action(dqn,
                                                         explorer,
                                                         state,
                                                         env.action_space.n)
            next_state, reward, done, _ = env.step(action)
            next_state = frame_stack.push_get(process_frame(next_state))
            episode_score += reward
            env.render()
            state = next_state
        print('episode score {}'.format(episode_score))


def display_processd_frame():
    env.reset()
    observation, reward, done, _ = env.step(env.action_space.sample())
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(observation)
    proc = process_frame(observation)
    ax[1].imshow(proc)
    plt.show()


def main():
    #train()
    play()
    #display_processd_frame()


if __name__ == '__main__':
    main()
