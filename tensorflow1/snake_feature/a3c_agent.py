'''
Author: Sunghoon Hong
Title: a3c_agent.py
Version: 0.0.1
Description: a3c Agent
Detail:
    Action size = 3
    Loss function of Critic = MSE
    Apply TD-lambda
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
from keras.models import Sequential, Model
from keras.layers import TimeDistributed, Reshape, Input, Reshape
from keras.layers import Conv2D, Flatten, Dense, GRU, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from sklearn.preprocessing import StandardScaler, scale
from PIL import ImageOps, Image
from snake_env import Env, ACTION

np.set_printoptions(precision=4, suppress=True)


def preprocess(observe):
    return observe / 20.

class A3CAgent:
    def __init__(self, state_size, action_size, seq_size, gamma,
                actor_lr, critic_lr, thread_num, lambd, tmax, entropy,
                verbose, load_model, render, save_rate, reward_clip):
        self.lock = threading.Lock()
        self.render = render
        self.save_rate = save_rate

        # hyperparameter
        self.seq_size = seq_size
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.threads = thread_num
        self.lambd = lambd
        self.tmax = tmax
        self.entropy = entropy

        self.state_size = state_size
        self.state_shape = [1] + self.state_size
        # self.history_size = [seq_size] + state_size
        self.action_size = action_size

        self.actor, self.critic = self.build_model()
        self.actor_update = self.actor_optimizer()
        self.critic_update = self.critic_optimizer()
        
        self.reward_clip = reward_clip
        self.stats = []

        if verbose:
            self.actor.summary()
            self.critic.summary()

        if load_model:
            self.load_model('./save_model/a3c')

    def train(self):
        agents = [Agent(
                    tid = tid,
                    state_size = self.state_size,
                    action_size = self.action_size,
                    seq_size = self.seq_size,
                    gamma = self.gamma,
                    lambd = self.lambd,
                    actor_lr = self.actor_lr,
                    critic_lr = self.critic_lr,
                    tmax = self.tmax,
                    model = [self.actor, self.critic],
                    optimizer = [self.actor_update, self.critic_update],
                    build_model = self.build_model,
                    reward_clip = self.reward_clip,
                    stats = self.stats,
                    render = self.render
                )
                for tid in range(self.threads)]

        for agent in agents:
            time.sleep(1)
            agent.start()

        print(dt.now().strftime('%Y-%m-%d %H:%M:%S'), '%s agents Ready!' % self.threads)
        
        highscore = 0

        global episode
        if episode > 100:
            if os.path.exists('a3c_output.csv'):
                with open('a3c_output.csv', 'r') as f:
                    read = csv.reader(f)
                    for i, line in enumerate(reversed(list(read))):
                        if i == 100:
                            highscore /= 100.
                            break
                        highscore += float(line[3])
        print('Highscore: %.3f' % highscore)
        while True:
            # for i in range(30):
            #     print('Next Update After %d (sec)' % (300-i*10), end='\r', flush=True)
            #     time.sleep(10)
            time.sleep(self.save_rate)
            if self.stats:
                with self.lock:
                    stats = deepcopy(self.stats)
                    self.stats.clear()
                with open('a3c_output.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    for row in stats:
                        wr.writerow(row)
                self.save_model('./save_model/a3c')
                mean = np.mean(np.float32(np.split(stats, [-1], axis=1)[0]), axis=0)
                if mean[3] > highscore:
                    highscore = mean[3]
                    self.save_model('./save_model/a3c_high')
                print('%s: %s Episodes Trained! AvgScore:%s AvgStep:%s AvgPmax:%s' 
                        % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), 
                        len(stats), mean[3], mean[1], mean[4]))
                stats = None
            else:
                print('%s: No Episodes...' % (dt.now().strftime('%Y-%m-%d %H:%M:%S')))

    def build_model(self):
        state = Input(shape=self.state_size)
        state_process = Dense(100, activation='elu')(state)
        state_process = BatchNormalization()(state_process)
        state_process = Dense(100, activation='elu')(state_process)
        state_process = BatchNormalization()(state_process)
        policy = Dense(self.action_size, activation='softmax', kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(state_process)
        value = Dense(1, activation='linear', kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(state_process)

        actor = Model(inputs=state, outputs=policy, name='Actor')
        critic = Model(inputs=state, outputs=value, name='Critic')

        actor._make_predict_function()
        critic._make_predict_function()

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

        loss = cross_entropy + self.entropy * entropy

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # MSE loss
        loss = 0.5 * K.mean(K.square(discounted_prediction - value))
        # # Huber loss
        # error = K.abs(discounted_prediction - value)
        # quadratic = K.clip(error, 0.0, 1.0)
        # linear = error - quadratic
        # loss = K.mean(0.5 * K.square(quadratic) + linear)

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_prediction],
                           [loss], updates=updates)
        return train

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

    def get_action(self, state):
        policy = self.actor.predict(state)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy


class Agent(threading.Thread):
    def __init__(self, tid, action_size, state_size, model, optimizer,
                lambd, gamma, seq_size, actor_lr, critic_lr, tmax,
                reward_clip, build_model, stats, render):
        threading.Thread.__init__(self)
        self.lock = threading.Lock()
        self.tid = tid
        self.action_size = action_size
        self.state_size = state_size
        self.seq_size = seq_size
        self.state_shape = [1] + state_size
        self.actor, self.critic = model
        self.actor_update, self.critic_update = optimizer
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lambd = lambd
        self.stats = stats
        self.reward_clip = reward_clip
        self.render = render and tid == 0

        self.histories, self.actions, self.rewards = [], [], []

        self.local_actor, self.local_critic = build_model()
        self.update_local_model()

        self.top_score = 0
        self.t_max = tmax
        self.t = 0
        # print('Agent %d Ready!' % self.tid)

    def run(self):
        global episode
        env = Env()
        while True:
            # print(self.tid, 'Still Training!')
            step = 0
            avg_p_max = 0
            reward_sum = 0
            actor_loss = 0
            critic_loss = 0
            done = False
            observe, _, _, _ = env.reset()

            state = preprocess(observe).reshape(self.state_shape)
            while not done:
                step += 1
                self.t += 1
                if self.render:
                    env.render()
                action, policy = self.get_action(state)
                real_action = action
                next_observe, reward, done, info = env.step(real_action)
                if self.reward_clip:
                    reward = np.clip(reward, -1.0, 1.0)
                next_state = preprocess(next_observe).reshape(self.state_shape)

                avg_p_max += np.amax(policy)
                reward_sum += reward
                self.append_sample(state, action, reward)

                state = next_state

                if self.t >= self.t_max or done:
                    self.t = 0
                    # if timeout, get returns with next pred value
                    mask = False if done and info == 'timeout' else done
                    actor_loss, critic_loss = self.train_model(next_state, mask)
                    actor_loss += actor_loss
                    critic_loss += critic_loss
                    self.update_local_model()

            episode += 1
            avg_p_max = avg_p_max / float(step)
            train_num = step // self.t_max + 1
            avg_actor_loss = actor_loss / train_num
            avg_critic_loss = critic_loss / train_num
            stats = [
                episode, step,
                reward_sum, env.game.score,
                avg_p_max, avg_actor_loss, avg_critic_loss,
                info
            ]
            with self.lock:
                self.stats.append(stats)

    def train_model(self, next_state, done):

        # compute Advantage with GAE and returns
        histories = np.zeros([len(self.histories) + 1] + self.state_size)
        for i in range(len(self.histories)):
            histories[i] = self.histories[i]
        histories[len(self.histories)] = next_state
        values = self.local_critic.predict(histories)
        values = np.reshape(values, len(values))

        # discounted prediction with TD lambda
        advantages = np.zeros_like(self.rewards)
        # returns = np.zeros_like(self.rewards)
        GAE = 0
        if done:
            values[-1] = np.float32([0])
        # G = values[-1]
        for t in reversed(range(0, len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t+1] - values[t]
            GAE = delta + self.gamma * self.lambd * GAE
            # returns[t] = G = self.rewards[t] + self.gamma * G
            advantages[t] = GAE

        returns = advantages + values[:-1]
        # advantage & returns regularization
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)
        # advantages = returns - values[:-1]
        [actor_loss] = self.actor_update([histories[:-1], self.actions, advantages])
        [critic_loss] = self.critic_update([histories[:-1], returns])
        self.histories, self.actions, self.rewards = [], [], []
        return actor_loss, critic_loss

    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def get_action(self, state):
        policy = self.local_actor.predict(state)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    def append_sample(self, state, action, reward):
        self.histories.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

if __name__ == "__main__":

    global episode
    episode = 0
    if not os.path.exists('save_graph/a3c'):
        os.makedirs('save_graph/a3c')
    if not os.path.exists('save_model'):
        os.makedirs('save_model')
    if os.path.exists('a3c_output.csv'):
        with open('a3c_output.csv', 'r') as f:
            read = csv.reader(f)
            episode = int(float(next(reversed(list(read)))[0]))
        print(episode)

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_rate',  type=int,   default=60,     help='Log and save model per save_rate (sec)')
    parser.add_argument('--threads',    type=int,   default=16,     help='The number of threads')
    parser.add_argument('--tmax',       type=int,   default=64,     help='Max length of trajectory')
    parser.add_argument('--lr',         type=float, default=1e-4,   help='Learning rate of critic. lr of actor will be divided by 10')
    parser.add_argument('--entropy',    type=float, default=1e-3,   help='Weight of entropy of actor loss (beta)')
    parser.add_argument('--lambd',      type=float, default=0.95,   help='TD(lambda). The bigger lambda, The bigger weight on future reward')
    parser.add_argument('--seqsize',    type=int,   default=1,      help='Length of sequence')
    parser.add_argument('--gamma',      type=float, default=0.99,   help='Discount factor')
    parser.add_argument('--reward_clip',action='store_true',        help='Reward will be clipped in [-1, 1]')
    parser.add_argument('--render',     action='store_true',        help='First agent render')
    parser.add_argument('--load_model', action='store_true',        help='Load model in ./save_model/')
    parser.add_argument('--verbose',    action='store_true',        help='Print summary of model of global network')
    args = parser.parse_args()
    
    env = Env()
    global_agent = A3CAgent(
        state_size = env.state_size,
        action_size = env.action_size,
        seq_size = args.seqsize,
        gamma = args.gamma,
        lambd = args.lambd,
        entropy = args.entropy,
        tmax = args.tmax,
        actor_lr = args.lr,
        critic_lr = args.lr,
        thread_num = args.threads,
        load_model=args.load_model,
        verbose=args.verbose,
        render=args.render,
        save_rate=args.save_rate,
        reward_clip=args.reward_clip
    )
    global_agent.train()