'''
Author: Sunghoon Hong
Title: debug_ra3c.py
Version: 0.0.1
Description: Debugging ra3c Agent
Detail:
    0.5 pmax action
'''

import os   
import time
import random
import threading
import csv
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import losses as tfloss
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Input, LSTM, TimeDistributed, Reshape
from keras.optimizers import Adam
from keras import backend as K
from PIL import ImageOps, Image
from matplotlib import pyplot as plt
from snake_env import Env, ACTION
# from ra3c_agent import A3CAgent

global episode
episode = 0
if not os.path.exists('save_graph/ra3c_agent'):
    os.makedirs('save_graph/ra3c_agent')
if not os.path.exists('save_model'):
    os.makedirs('save_model')
if os.path.exists('ra3c_output.csv'):
    with open('ra3c_output.csv', 'r') as f:
        read = csv.reader(f)
        episode = int(float(next(reversed(list(read)))[0]))
    print(episode)
    
EPISODES = 8000000

SAVE_STAT_EPISODE_RATE = 100
SAVE_STAT_TIME_RATE = 600 #sec
THREAD_NUM = 64
RESIZE = 84
SAVE_GRAPH = True
LOAD_MODEL = True
VERBOSE = False

def preprocess(observe):
    ret = Image.fromarray(observe)
    ret = ImageOps.mirror(ret.rotate(270)).convert('L').resize((RESIZE, RESIZE))
    return np.asarray(ret)


class A3CAgent:
    def __init__(self, verbose=False, load_model=True, render=False):
        self.verbose = verbose
        self.render = render
        # hyperparameter
        self.seq_size = 2
        self.discount = 0.99
        self.no_op_steps = 40
        self.actor_lr = 5e-4
        self.critic_lr = 5e-4
        self.threads = THREAD_NUM

        self.state_size = (self.seq_size, RESIZE, RESIZE)
        self.action_size = 4

        self.actor, self.critic = self.build_model()
        
        self.stats = []

        if load_model:
            self.load_model('./save_model/ra3c')

    def build_model(self):
        state_size = list(self.state_size)
        state_size.append(1)
        input = Input(shape=self.state_size)
        reshape = Reshape(state_size)(input)

        conv = TimeDistributed(Conv2D(16, (8, 8), strides=(4, 4), activation='relu', kernel_initializer='he_normal'))(reshape)
        conv = TimeDistributed(Conv2D(32, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='he_normal'))(conv)
        conv = TimeDistributed(Flatten())(conv)
        lstm = LSTM(256, activation='tanh', kernel_initializer='he_normal')(conv)

        policy = Dense(self.action_size, activation='softmax', kernel_initializer='he_normal')(lstm)
        value = Dense(1, activation='linear', kernel_initializer='he_normal')(lstm)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()
        
        if self.verbose:
            actor.summary()
            critic.summary()

        return actor, critic

    def load_model(self, name):
        if os.path.exists(name + '_actor.h5'):
            self.actor.load_weights(name + '_actor.h5')
            print('Actor loaded')
        if os.path.exists(name + '_critic.h5'):
            self.critic.load_weights(name + '_critic.h5')
            print('Critic loaded')

    def get_action(self, history):
        policy = self.actor.predict(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    def play(self):
        env = Env()
        if self.render:
            env.init_render()
        scores = 0
        steps = 0
        episodes = 10
        for e in range(episodes):
            step = 0
            done = False
            observe, _, _, _ = env.reset()
            state = preprocess(observe).reshape((1, RESIZE, RESIZE))
            state = np.float32(state / 255.)
            history = np.copy(state)
            for _ in range(self.seq_size - 1):
                history = np.append(history, state, axis=0)
                state = np.copy(state)
            history = np.reshape([history], (1, self.seq_size, RESIZE, RESIZE))
            while not done:
                # snap1 = history[0][0]
                # snap2 = history[0][1]
                # Image.fromarray(snap1 * 255.).show()
                # Image.fromarray(snap2 * 255.).show()
                step += 1
                if self.render:
                    env.render()
                action, policy = self.get_action(history)
                pmax_action = np.argmax(policy)
                print(ACTION[action], policy, ACTION[pmax_action])
                while True:
                    key = input('Press y or action: ')
                    if key in ['0', '1', '2', '3']:
                        action = int(key)
                        break
                    elif key == 'y':
                        break
                # if np.random.uniform() > 0.5:
                #     action = pmax_action
                next_observe, reward, done, info = env.step(action+1)
                next_state = preprocess(next_observe)
                next_state = np.reshape([next_state], (1, 1, RESIZE, RESIZE))
                next_state = np.float32(next_state / 255.)
                next_history = np.append(next_state, history[:, :(self.seq_size-1), :, :], axis=1)

                history = next_history

            steps += step
            scores += env.game.score
            step = 0

        print('AvgScore: %s AvgStep: %s' % (scores/episodes, steps/episodes))
        return scores/episodes, steps/episodes


if __name__ == "__main__":
    global_agent = A3CAgent(
        load_model=LOAD_MODEL,
        verbose=VERBOSE,
        render=True
    )
    global_agent.play()
