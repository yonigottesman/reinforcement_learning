import os
import random

import numpy as np
import tensorflow as tf
import gym

from Explorer import Explorer
from memory import fill_memory, ReplayMemory, Experience

env = gym.make('CartPole-v1')
PATH = 'models/cartpole.h5'
if not os.path.exists('models'):
    os.makedirs('models')


class DQN():
    def __init__(self, state_shape, n_actions, lr=0.001, saved_model=None):
        if saved_model:
            self.model = tf.keras.models.load_model(saved_model)
        else:
            state_input = tf.keras.layers.Input(shape=state_shape)
            action_input = tf.keras.layers.Input(shape=n_actions)
            a1 = tf.keras.layers.Dense(24, activation='relu')(state_input)
            a2 = tf.keras.layers.Dense(24, activation='relu')(a1)
            a3 = tf.keras.layers.Dense(n_actions)(a2)
            output = tf.keras.layers.Multiply()([a3, action_input])
            self.model = tf.keras.models.Model(inputs=[state_input, action_input],
                                               outputs=output)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.model.compile(loss='mse', optimizer=optimizer)


def predict_action(dqn, explorer, state, n_actions):

    if explorer.explore():
        # exploration
        action = random.randint(0, n_actions-1)
    else:
        # exploitation
        qs = dqn.predict([state.reshape((1, *state.shape)),
                                np.ones((1, n_actions))])
        action = np.argmax(qs)

    return action, explorer.explore_prob()


# training hyperparams
memory_size = 20000
batch_size = 64
lr = 0.001
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.0001  # exponential decay rate for exploration prob
gamma = 0.9  # Discounting rate


def learn(model, memory):

    batch = memory.sample(batch_size)
    states_mb = tf.convert_to_tensor(batch.state)
    actions_mb = tf.one_hot(tf.convert_to_tensor(batch.action),env.action_space.n)
    rewards_mb = tf.convert_to_tensor(batch.reward)
    next_states_mb = tf.convert_to_tensor(batch.next_state)
    dones_mb = tf.convert_to_tensor(batch.done)

    #TODO - predict only non final to save some time
    qs_next_state = model.predict([next_states_mb,
                                       tf.ones(actions_mb.shape)])

    target_qs_batch = tf.where(dones_mb,
                                   rewards_mb,
                                   rewards_mb + gamma * tf.reduce_max(qs_next_state,axis=1))

    history = model.fit([states_mb, actions_mb],
                        actions_mb * target_qs_batch[:, None],
                        epochs=1, batch_size=len(batch), verbose=0)
    return history


def train():
    memory = ReplayMemory(memory_size)
    fill_memory(memory, env, batch_size)
    explorer = Explorer(explore_start, explore_stop, decay_rate)
    dqn = DQN(state_shape=env.observation_space.shape,
              n_actions=(env.action_space.n), lr=lr)
    dqn.model.summary()

    rewards_list = []
    loss = 1
    for episode in range(5000):
        episode_rewards = 0
        state = env.reset()
        done = False
        while not done:
            action, explore_probability = predict_action(dqn.model,
                                                         explorer,
                                                         state,
                                                         env.action_space.n)

            next_state, reward, done, _ = env.step(action)
            # env.render()
            episode_rewards += reward
            memory.push(state, action, reward, next_state, done)
            state = next_state
            loss = learn(dqn.model, memory).history['loss']

            if done:
                rewards_list.append(episode_rewards)
                moving_average = np.mean(rewards_list[-100:])
                if episode % 10 == 0:
                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(episode_rewards),
                          'Explore P: {:.4f}'.format(explore_probability),
                          'Training Loss {}'.format(loss),
                          'Moving average {}'.format(moving_average))

        if episode % 10 == 0:
            dqn.model.save(PATH)


def play():
    dqn = DQN(state_shape=None, n_actions=None, lr=0, saved_model=PATH)
    n_actions = env.action_space.n
    for episode in range(5000):
        state = env.reset()
        done = False
        episode_score = 0
        while not done:
            qs = dqn.model.predict([state.reshape((1, *state.shape)),
                                    np.ones((1, n_actions))])
            action = np.argmax(qs)
            next_state, reward, done, _ = env.step(action)
            episode_score += reward
            env.render()
            state = next_state
        print('episode score {}'.format(episode_score))


def main():
    train()
    #play()


if __name__ == '__main__':
    main()
