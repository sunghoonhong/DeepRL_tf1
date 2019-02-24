'''
Author: Sunghoon Hong
Title: dqn_agent.py
Version: 0.0.1
Description: Deep Q-Network Agent
CNN + history + Experience Replay + Target Model
'''

import os
import csv
import time
import random
from collections import deque
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, TimeDistributed, LSTM, Reshape
from keras.optimizers import Adam
from keras import backend as K
from PIL import ImageOps
from PIL import Image
from matplotlib import pyplot as plt
from snake_env import Env, ACTION

RESIZE = 84
# EPISODES = 20000
# SAVE_STAT_TIME_RATE = 300 #sec
SAVE_EPISODE_RATE = 100

TRAIN = True
LOAD_MODEL = True
VERBOSE = False


class DQNAgent:

    def __init__(self, render=False, load_model=False,
                verbose=False):
        self.render = render
        self.load_model = load_model
        self.verbose = verbose
        self.action_size = 4
        self.seq_size = 2
        self.state_size = (self.seq_size, RESIZE, RESIZE)

        # hyperparameter
        self.discount = 0.99
        self.lr = 1e-5
        self.epsilon = 1.0
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000
        self.epsilon_decay_step = (
            (self.epsilon_start - self.epsilon_end) / self.exploration_steps
        )
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        
        self.memory = deque(maxlen=100000)
        # self.no_op_steps = 30
        
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.opt = self.build_optimizer()

        if self.load_model:
            if os.path.exists('./save_model/dqn.h5'):
                self.model.load_weights('./save_model/dqn.h5')
                self.target_model.load_weights('./save_model/dqn.h5')
                print('Using Model ./save_model/dqn.h5...')
            else:
                print('No model exists.')

    def build_optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        Q = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - Q)

        quadratic = K.clip(error, 0.0, 1.0)
        linear = error - quadratic
        loss = K.mean(0.5 * K.square(quadratic) + linear)

        optimizer = Adam(lr=self.lr, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)
        return train

    def build_model(self):
        state_size = list(self.state_size)
        state_size.append(1)
        input = Input(shape=self.state_size)
        reshape = Reshape(state_size)(input)
        conv = TimeDistributed(Conv2D(16, (8, 8), strides=(4, 4), activation='relu', kernel_initializer='he_normal'))(reshape)
        conv = TimeDistributed(Conv2D(32, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='he_normal'))(conv)
        conv = TimeDistributed(Flatten())(conv)
        lstm = LSTM(512, activation='tanh', kernel_initializer='he_normal')(conv)
        Qvalue = Dense(self.action_size, activation='linear', kernel_initializer='he_normal')(lstm)

        model = Model(inputs=input, outputs=Qvalue)
        if self.verbose:
            model.summary()
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, history):
        Q = self.model.predict(history)
        action = np.argmax(Q[0])
        Q_max = Q[0][action]
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        return action, Q_max

    def append_sample(self, history, action, reward, next_history, done):
        self.memory.append((history, action, reward, next_history, done))

    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step
        mini_batch = random.sample(self.memory, self.batch_size)
#         mini_batch = np.transpose(mini_batch)
        history = np.zeros((self.batch_size, self.state_size[0],
                        self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                        self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size, 1))
        actions = np.zeros((self.batch_size, 1))
        rewards = np.zeros((self.batch_size, 1))
        dones = np.zeros((self.batch_size, 1))

        for i in range(self.batch_size):
          history[i] = np.float32(mini_batch[i][0])
          next_history[i] = np.float32(mini_batch[i][3])
          actions[i] = np.float32(mini_batch[i][1])
          rewards[i] = np.float32(mini_batch[i][2])
          dones[i] = np.float32(mini_batch[i][4])
          
        target_val = self.target_model.predict(next_history)        
        target = rewards + (1-dones) * self.discount * np.amax(target_val, axis=1).reshape(self.batch_size, 1)

        # for i in range(self.batch_size):
        #     if dones[i]:
        #         target[i] = rewards[i]
        #     else:
        #         target[i] = (
        #             rewards[i] + self.discount * np.amax(target_val[i])
        #         )

        self.opt([history, actions, target])


def preprocess(observe):
    ret = Image.fromarray(observe)
    ret = ImageOps.mirror(ret.rotate(270)).convert('L').resize((RESIZE, RESIZE))
    return np.asarray(ret)


if __name__ == '__main__':
    env = Env()
    
    if not os.path.exists('./save_graph/dqn_agent'):
        os.makedirs('./save_graph/dqn_agent')
    if not os.path.exists('./save_model'):
        os.makedirs('./save_model')

    if TRAIN:
        agent = DQNAgent(load_model=LOAD_MODEL)
        stats = []
        global_step = 0
        
        episode = 0
        if os.path.exists('./dqn_output.csv'):
            with open('./dqn_output.csv', 'r') as f:
                read = csv.reader(f)
                episode = int(float(next(reversed(list(read)))[0]))
        print(episode)

        while True:
            done = False
            step = 0
            reward_sum = 0
            avg_Q_max = 0

            observe, _, _, _ = env.reset()

            state = preprocess(observe).reshape((1, RESIZE, RESIZE))
            state = np.float32(state / 255.)
            history = np.copy(state)
            for _ in range(agent.seq_size - 1):
                history = np.append(history, state, axis=0)
                state = np.copy(state)
            history = np.reshape([history], (1, agent.seq_size, RESIZE, RESIZE))

            while not done:
                if agent.render:
                    env.render()
                step += 1
                global_step += 1

                action, Q_max = agent.get_action(history)
                real_action = action + 1
                next_observe, reward, done, info = env.step(real_action)
                next_state = preprocess(next_observe)
                next_state = np.reshape([next_state], (1, 1, RESIZE, RESIZE))
                next_state = np.float32(next_state / 255.)
                next_history = np.append(next_state, history[:, :(agent.seq_size-1), :, :], axis=1)

                agent.append_sample(history, action, reward, next_history, done)

                if len(agent.memory) >= agent.train_start:
                    agent.train_model()

                if global_step % agent.update_target_rate == 0:
                    agent.update_target_model()

                history = next_history

                # statistics
                reward_sum += reward
                avg_Q_max += Q_max

            # stats.append((e, env.game.score, step, info))
            episode += 1
            stats.append([episode, step, reward_sum, env.game.score, avg_Q_max/step, info])

            if episode % SAVE_EPISODE_RATE == 0:
                with open('dqn_output.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    for row in stats:
                        wr.writerow(row)
                mean = np.mean(np.float32(np.transpose(stats)[:5]), axis=1)
                print('Ep %s Step: %s Score: %s Qmax: %s' % (episode, mean[1], mean[3], mean[4]), end='\r')
                stats.clear()
                agent.model.save_weights('./save_model/dqn.h5')
                # print('[Episode %s] AvgScore: %s, AvgStep: %s' % (e, np.mean(scores[e-100:]), np.mean(steps[e-100:])))
    else:
        agent = DQNAgent(render=True, load_model=LOAD_MODEL)
        for e in range(10):
            done = False
            observe, _, _, _ = env.reset()

            state = preprocess(observe).reshape((1, RESIZE, RESIZE))
            state = np.float32(state / 255.)
            history = np.copy(state)
            for _ in range(agent.seq_size - 1):
                history = np.append(history, state, axis=0)
                state = np.copy(state)
            history = np.reshape([history], (1, agent.seq_size, RESIZE, RESIZE))

            while not done:
                if agent.render:
                    env.render()
                step += 1
                global_step += 1

                action, Q_max = agent.get_action(history)
                real_action = action + 1
                next_observe, reward, done, info = env.step(real_action)
                next_state = preprocess(next_observe)
                next_state = np.reshape([next_state], (1, 1, RESIZE, RESIZE))
                next_state = np.float32(next_state / 255.)
                next_history = np.append(next_state, history[:, :(agent.seq_size-1), :, :], axis=1)
    
                history = next_history

            print('[Episode %s] Score: %s, Step: %s, Info: %s' % (e, env.game.score, step, info))