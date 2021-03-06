'''
Author: Sunghoon Hong
Title: racer_agent.py
Version: 0.0.1
Description: RACER Agent
Detail:
    Sequence size = 4
    Action size = 3
    RESIZE = 20
    Loss function of Critic = MSE
    Apply TD-lambda(1)
'''

import os   
import time
import random
import threading
import csv
import argparse
from copy import deepcopy
from datetime import datetime as dt
from collections import deque
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import TimeDistributed, Reshape, Input
from keras.layers import Conv2D, Flatten, Dense, GRU, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from PIL import ImageOps, Image
from snake_env import Env, ACTION

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

parser = argparse.ArgumentParser()
parser.add_argument('--save_rate', type=int, default=300)
parser.add_argument('--threads', type=int, default=32)
parser.add_argument('--tmax', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--entropy', type=float, default=0.01)
parser.add_argument('--lambd', type=float, default=1)
parser.add_argument('--seqsize', type=int, default=4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--reward', type=str, default='default')
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

np.set_printoptions(precision=4, suppress=True)

SAVE_STAT_TIME_RATE = args.save_rate #sec
THREAD_NUM = args.threads
RESIZE = 20
TRAIN = True
# TRAIN = False
LOAD_MODEL = True
VERBOSE = False

# Hyper Parameter
K_STEP = args.tmax
ENT_WEIGHT = args.entropy
LR = args.lr
LAMBDA = args.lambd
SEQ_SIZE = args.seqsize
GAMMA = args.gamma
REWARD_CLIP = args.reward

def preprocess(observe):
    ret = Image.fromarray(observe)
    ret = ImageOps.mirror(ret.rotate(270)).convert('L').resize((RESIZE, RESIZE))
    return np.asarray(ret)


class A3CAgent:
    def __init__(self, verbose=False, load_model=True, render=False):
        self.verbose = verbose
        self.render = render

        # hyperparameter
        self.seq_size = SEQ_SIZE
        self.gamma = GAMMA
        self.actor_lr = LR
        self.critic_lr = LR
        self.threads = THREAD_NUM
        self.lambd = LAMBDA
        self.batch_size = 256
        self.memory_size = 1000000
        self.train_start = 50000
        self.actor_rate = 10

        self.state_size = (self.seq_size, RESIZE, RESIZE, 1)
        self.action_size = 3

        self.actor, self.critic = self.build_model()
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]
        
        self.stats = []
        self.memory = deque(maxlen=self.memory_size)

        if load_model:
            self.load_model('./save_model/ra3c')

    def train(self):
        agents = [Agent(tid, self.action_size, self.state_size,
                        [self.actor, self.critic], self.optimizer,
                        self.lambd, self.gamma, self.seq_size, self.build_model,
                        self.stats, self.memory)
                  for tid in range(self.threads)]

        for agent in agents:
            time.sleep(1)
            agent.start()
        print(dt.now().strftime('%Y-%m-%d %H:%M:%S'), '%s agents Ready!' % self.threads)
        
        T = 0
        while True:
            T += 1
            ## log
            if len(self.stats) > 100:
                stats = deepcopy(self.stats)
                self.stats.clear()
                with open('racer_output.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    for row in stats:
                        wr.writerow(row)
                self.save_model('./save_model/racer')
                mean = np.mean(np.float32(np.split(stats, [-1], axis=1)[0]), axis=0)
                print('%s: %s Episodes Trained! AvgScore:%s AvgStep:%s AvgPmax:%s' 
                        % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), 
                        len(stats), mean[3], mean[1], mean[4]))
                stats = None
            # else:
            #     print('%s: No Episodes...' % (dt.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            ## train
            if len(self.memory) > self.train_start:
                self.train_batch()
            if T >= self.actor_rate:
                T = 0
                for agent in agents:
                    agent.update_local_model()

    def build_model(self):
        input = Input(shape=self.state_size)
        # conv = TimeDistributed(Conv2D(64, (8, 8), strides=(4, 4), padding='same', activation='elu', kernel_initializer='he_normal'))(input)
        # conv = TimeDistributed(Conv2D(32, (4, 4), strides=(2, 2), activation='elu', kernel_initializer='he_normal'))(conv)
        # conv = TimeDistributed(Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal'))(conv)
        # conv = TimeDistributed(Conv2D(16, (1, 1), activation='elu', kernel_initializer='he_normal'))(conv)
        conv = TimeDistributed(Flatten())(input)
        batch_norm = BatchNormalization()(conv)
        gru = GRU(256, activation='tanh', kernel_initializer='he_normal')(batch_norm)
        policy = Dense(self.action_size, activation='softmax', kernel_initializer='he_normal')(gru)
        value = Dense(1, activation='linear', kernel_initializer='he_normal')(gru)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()
        
        if self.verbose:
            actor.summary()
            critic.summary()

        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.mean(cross_entropy)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.mean(entropy)

        loss = cross_entropy + ENT_WEIGHT * entropy

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # MSE loss
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_prediction],
                           [loss], updates=updates)
        return train

    
    def train_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.zeros([None] + list(self.state_size))
        actions = np.zeros([None] + list(self.action_size))
        adv_actor = np.zeros([None, 1])
        adv_critic = np.zeros([None, 1])

        for transition in batch:
            s = transition[0]
            a = transition[1]
            r = transition[2]
            d = transition[3]
            
            values = self.critic.predict(s)
            discounted = np.zeros_like(r)
            G = 0 if d else values[-1]
            for t in reversed(range(r)):
                G = r[t] + self.gamma * G
                discounted[t] = G
            advantage = discounted - values[:-1]

            np.append(states, s[:-1], axis=0)
            np.append(actions, a, axis=0)
            np.append(adv_actor, advantage, axis=0)
            np.append(adv_critic, discounted, axis=0)
        
        actor_loss = self.optimizer[0]([states, actions, adv_actor])
        critic_loss = self.optimizer[1]([states, adv_critic])
        return actor_loss, critic_loss

    def load_model(self, name):
        if os.path.exists(name + '_actor.h5'):
            self.actor.load_weights(name + '_actor.h5')
            print('Actor loaded')
        if os.path.exists(name + '_critic.h5'):
            self.critic.load_weights(name + '_critic.h5')
            print('Critic loaded')

    def save_model(self, name):
        self.actor.save_weights(name + '_actor.h5')
        self.critic.save_weights(name + '_critic.h5')

    def get_action(self, history):
        policy = self.actor.predict(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    def play(self, episodes=10, delay=0, improve='policy', debug=False, SNAPSHOT=False):
        env = Env()
        scores = 0
        steps = 0
        print('Value\tRandom\tGreedy\tPolicy')
        for e in range(episodes):
            step = 0
            done = False
            observe, _, _, _ = env.reset()
            state = preprocess(observe).reshape((1, RESIZE, RESIZE, 1))
            state = np.float32(state / 255.)
            history = np.stack([state] * self.seq_size, axis=1)
            while not done:
                time.sleep(delay)
                step += 1
                if self.render:
                    env.render()
                    if SNAPSHOT:
                        snapshot = np.array([]).reshape([0,RESIZE])
                        for snap in history[0]:
                            snapshot = np.append(snapshot, snap, axis=0)
                        Image.fromarray(snapshot*255.).show()
                action, policy = self.get_action(history)
                if improve == 'greedy':
                    real_action = int(np.argmax(policy))
                elif improve == 'e-greedy':
                    real_action = int(np.argmax(policy)) if np.random.uniform(low=0.0, high=1.0) > 0.1 else action 
                else:
                    real_action = action
                value = self.critic.predict(history)
                print(value, '\t', ACTION[action], '\t', ACTION[int(np.argmax(policy))], '\t', policy)
                if debug:
                    while True:
                        a = input('Press y or action(w(stay), a(left), d(right)):')
                        if a=='y':
                            break
                        elif a=='w':
                            real_action = 0
                            break
                        elif a=='a':
                            real_action = 1
                            break
                        elif a=='d':
                            real_action = 2
                            break
                next_observe, reward, done, info = env.step(real_action)
                next_state = preprocess(next_observe).reshape((1, RESIZE, RESIZE, 1))
                next_state = np.float32(next_state / 255.)
                next_history = np.append(history[0][1:], next_state, axis=0)
                next_history = np.float32([next_history])
                history = next_history

            steps += step
            scores += env.game.score

            print('Score: %s Step: %s' % (env.game.score, step))
        return scores/episodes, steps/episodes


class Agent(threading.Thread):
    def __init__(self, tid, action_size, state_size, model,
                 optimizer, lambd, gamma, seq_size,
                 build_model, stats, memory):
        threading.Thread.__init__(self)

        self.tid = tid
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.seq_size = seq_size
        self.stats = stats
        self.lambd = lambd
        self.memory = memory

        self.render = args.render and tid == 0

        self.states, self.actions, self.rewards = [], [], []

        self.local_actor, self.local_critic = build_model()
        self.update_local_model()

        self.reward_sum = 0
        self.avg_p_max = 0
        self.actor_loss = 0
        self.critic_loss = 0

        self.top_score = 1
        self.t_max = K_STEP
        self.t = 0
        # print('Agent %d Ready!' % self.tid)

    def run(self):
        global episode
        env = Env()

        while True:
            # print(self.tid, 'Still Training!')
            step = 0
            self.avg_p_max = 0
            self.reward_sum = 0
            self.actor_loss = 0
            self.critic_loss = 0
            done = False
            observe, _, _, _ = env.reset()

            state = preprocess(observe).reshape((1, RESIZE, RESIZE, 1))
            state = np.float32(state / 255.)
            history = np.stack([state] * self.seq_size, axis=1)

            while not done:
                step += 1
                self.t += 1
                if self.render:
                    env.render()
                action, policy = self.get_action(history)
                real_action = action
                next_observe, reward, done, info = env.step(real_action)
                if REWARD_CLIP == 'clip':
                    reward = np.clip(reward, -1.0, 1.0)
                next_state = preprocess(next_observe).reshape((1, RESIZE, RESIZE, 1))
                next_state = np.float32(next_state / 255.)
                next_history = np.append(history[0][1:], next_state, axis=0)
                next_history = np.float32([next_history])

                self.avg_p_max += np.amax(policy)
                self.reward_sum += reward
                self.append_sample(history, action, reward)

                history = next_history

                if self.t >= self.t_max or done:
                    self.t = 0
                    actor_loss, critic_loss = self.upload_sample(next_history, done)
                    self.actor_loss += abs(actor_loss[0])
                    self.critic_loss += abs(critic_loss[0])

            episode += 1
            avg_p_max = self.avg_p_max / float(step)
            train_num = step // self.t_max + 1
            avg_actor_loss = self.actor_loss / train_num
            avg_critic_loss = self.critic_loss / train_num
            stats = [
                episode, step,
                self.reward_sum, env.game.score,
                avg_p_max, avg_actor_loss, avg_critic_loss,
                info
            ]
            self.stats.append(stats)

    def upload_sample(self, next_history, done):
        states = np.zeros((len(self.states) + 1, self.seq_size, RESIZE, RESIZE, 1))
        for i in range(len(self.states)):
            states[i] = self.states[i]
        states[len(self.states)] = next_history
        
        self.memory.append((states, self.actions, self.rewards, done))

    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def get_action(self, history):
        policy = self.local_actor.predict(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

if __name__ == "__main__":
    if TRAIN:            
        global_agent = A3CAgent(
            load_model=LOAD_MODEL,
            verbose=VERBOSE,
            render=False
        )
        global_agent.train()
        print('Train Start!')
    else:

        global_agent = A3CAgent(
            load_model=LOAD_MODEL,
            verbose=VERBOSE,
            render=True
        )
        global_agent.play()
