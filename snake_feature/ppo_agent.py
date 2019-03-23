
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

np.set_printoptions(suppress=True, precision=3)

class PPOAgent:
    def __init__(self, state_size, action_size, seq_size,
                gamma, lambd, horizon, entropy, actor_lr, critic_lr, thread_num, 
                reward_clip, clip, batch_size, epoch,
                verbose, load_model, render, save_rate, debug, timeout):
        # self.lock = threading.Lock()
        self.render = render
        self.save_rate = save_rate
        self.debug = debug
        self.timeout = timeout

        # hyperparameter
        self.seq_size = seq_size
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.threads = thread_num
        self.lambd = lambd
        self.horizon = horizon
        self.entropy = entropy
        self.clip = clip
        self.batch_size = batch_size
        self.epoch = epoch

        self.state_size = state_size
        self.state_shape = [1] + state_size
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
            self.load_model('./save_model/ppo')

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
        old_pi = K.placeholder(shape=[None, ])
        advantages = K.placeholder(shape=[None, ])
        policy = self.actor.output

        pi = K.sum(action * policy, axis=1)
        log_pi = K.log(pi + 1e-10)
        log_old_pi = K.log(old_pi + 1e-10)
        ratio = K.exp(log_pi - log_old_pi)
        cliped_ratio = K.clip(ratio, 1 - self.clip, 1 + self.clip)
        returns = K.minimum(ratio * advantages, cliped_ratio * advantages)
        returns = -K.mean(returns)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.mean(entropy)

        loss = returns + self.entropy * entropy

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, old_pi, advantages],
                           [loss, pi, ratio, cliped_ratio], updates=updates)
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

    def train(self):
        global episode
        highscore = 0

        if os.path.exists('ppo_highscore.csv'):
            with open('ppo_highscore.csv', 'r') as f:
                read = csv.reader(f)
                highscore = float(next(reversed(list(read)))[1])
        print('Highscore: %.3f' % highscore)

        env = Env()

        self.states = np.zeros([0] + self.state_size)
        self.actions = np.zeros([0] + [self.action_size])
        self.rewards, self.old_pi, self.advantages, self.returns = [], [], [], []

        actor_loss = 0
        critic_loss = 0
        t = 0
        self.t = 0
        while True:
            done = False
            step = 0
            reward_sum = 0
            pmax = 0
            timeout = 0
            observe, _, _, _ = env.reset()
            state = observe.reshape(self.state_shape) / 20.
            while not done and timeout < self.timeout:
                if self.render:
                    env.render()
                action, policy = self.get_action(state)
                if self.debug:
                    print(state, policy)
                    while True:
                        a = input('Press 0 1 2: ')
                        if a in ['0', '1', '2']:
                            action = int(a)
                            break
                    
                next_observe, reward, done, info = env.step(action)
                next_state = next_observe.reshape(self.state_shape) / 20.
                
                self.append_sample(state, action, reward, policy[action])

                timeout = 0 if info == 'goal' else timeout + 1
                step += 1
                t += 1
                pmax += np.amax(policy)
                reward_sum += reward

                state = next_state

            if not done:
                info = 'timeout'
            self.get_gae(next_state, done)
            self.t = t
            episode += 1

            avg_pmax = pmax / float(step)
            stats = [
                episode, step,
                reward_sum, env.game.score,
                avg_pmax, actor_loss, critic_loss,
                info
            ]
            self.stats.append(stats)

            if t >= self.horizon:
                actor_loss, critic_loss = self.train_model()
                t = 0
                self.t = 0
                # actor_loss += a_loss
                # critic_loss += c_loss
                if len(self.stats) >= self.save_rate:
                    with open('ppo_output.csv', 'a', encoding='utf-8', newline='') as f:
                        wr = csv.writer(f)
                        for row in self.stats:
                            wr.writerow(row)
                    self.save_model('./save_model/ppo')
                    mean = np.mean(np.float32(np.split(self.stats, [-1], axis=1)[0]), axis=0)
                    if mean[3] > highscore:
                        highscore = mean[3]
                        with open('ppo_highscore.csv', 'a', encoding='utf-8', newline='') as f:
                            wr = csv.writer(f)
                            wr.writerow([episode, highscore, dt.now().strftime('%Y-%m-%d %H:%M:%S')])
                        self.save_model('./save_model/ppo_high')
                    print('%s: %s Episodes Trained! Reward:%.3f Score:%.3f Step:%.3f Pmax:%.3f' 
                            % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), 
                            len(self.stats), mean[2], mean[3], mean[1], mean[4]))
                    self.stats.clear()

    def get_gae(self, next_state, done):
        states = np.append(self.states[self.t:], next_state, axis=0)
        values = self.critic.predict(states)
        values = np.reshape(values, len(values))
        
        # print(self.t, states.shape, values.shape)
        rewards = self.rewards[self.t:]
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        GAE = 0
        if done:
            values[-1] = np.float32([0])
        G = values[-1]
        
        for t in reversed(range(0, len(rewards))):
            delta = self.rewards[t] + self.gamma * values[t+1] - values[t]
            GAE = delta + self.gamma * self.lambd * GAE
            # returns[t] = G = self.rewards[t] + self.gamma * G
            advantages[t] = GAE
        returns = advantages + values[:-1]
        if self.debug:
            print('GET GAE!')
            print('state', states)
            print('value', values)
            print('return', returns)
            print('adv', advantages)
        # advantage regularization
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)
        if self.debug:
            print('reg ret', returns)
            print('reg adv', advantages)
        self.advantages += advantages.tolist()
        self.returns += returns.tolist()

    def train_model(self):
        n = len(self.states)
        arr = np.arange(n)
        self.advantages = np.array(self.advantages)
        self.returns = np.array(self.returns)
        self.old_pi = np.array(self.old_pi)

        if self.debug:
            print('Train model', n, 'sample')
            print('state', self.states)
            print('action', self.actions)
            print('old pi', self.old_pi)
            print('rewards', self.rewards)
            print('return', self.returns)
            print('advantage', self.advantages)
            
        actor_loss = 0
        critic_loss = 0
        batch_num  = n // self.batch_size
        for epoch in range(self.epoch):
            np.random.shuffle(arr)
            if self.debug:
                print(epoch, 'epoch ...')
            for i in range(batch_num):
                idx = arr[i * self.batch_size : (i+1) * self.batch_size]
                states = self.states[idx]
                actions = self.actions[idx]
                old_pi = self.old_pi[idx]
                advantages = self.advantages[idx]
                returns = self.returns[idx]
                [a_loss, pi, ratio, cliped] = self.actor_update([states, actions, old_pi, advantages])
                [c_loss] = self.critic_update([states, returns])
                actor_loss += a_loss
                critic_loss += c_loss
                if self.debug:
                    print(i, 'mini batch...')
                    print('idx', idx)
                    print('state', states)
                    print('action', actions)
                    print('old pi', old_pi)
                    print('pi', pi)
                    print('ratio', ratio)
                    print('cliped', cliped)
                    print('return', returns)
                    print('advantage', advantages)

        self.states = np.zeros([0] + self.state_size)
        self.actions = np.zeros([0] + [self.action_size])
        self.rewards, self.old_pi, self.advantages, self.returns = [], [], [], []
        
        return actor_loss / (epoch * batch_num), critic_loss  / (epoch * batch_num)

    def get_action(self, state):
        policy = self.actor.predict(state)[0]
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action, policy

    def append_sample(self, state, action, reward, pi):
        self.states = np.append(self.states, state, axis=0)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions = np.append(self.actions, [act], axis=0)
        self.rewards.append(reward)
        self.old_pi.append(pi)

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

if __name__ == "__main__":

    global episode
    episode = 0
    if not os.path.exists('save_graph/ppo_agent'):
        os.makedirs('save_graph/ppo_agent')
    if not os.path.exists('save_model'):
        os.makedirs('save_model')
    if os.path.exists('ppo_output.csv'):
        with open('ppo_output.csv', 'r') as f:
            read = csv.reader(f)
            episode = int(float(next(reversed(list(read)))[0]))
        print('Last Episode:', episode)

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_rate',  type=int,   default=100,     help='Log and save model per save_rate (sec)')
    parser.add_argument('--threads',    type=int,   default=16,     help='The number of threads')
    parser.add_argument('--lr',         type=float, default=1e-4,   help='Learning rate of critic. lr of actor will be divided by 10')
    parser.add_argument('--entropy',    type=float, default=1e-3,   help='Weight of entropy of actor loss (beta)')
    parser.add_argument('--gamma',      type=float, default=0.99,   help='Discount factor')
    parser.add_argument('--lambd',      type=float, default=0.95,   help='TD(lambda). The bigger lambda, The bigger weight on future reward')
    parser.add_argument('--batch_size', type=int,   default=16,     help='Mini-batch size')
    parser.add_argument('--horizon',    type=int,   default=256,    help='Time horizon')
    parser.add_argument('--seqsize',    type=int,   default=1,      help='Length of sequence')
    parser.add_argument('--epoch',      type=int,   default=3,      help='Update epochs')
    parser.add_argument('--clip',       type=float, default=0.2,    help='Clip ratio')
    parser.add_argument('--timeout',    type=int,   default=400,    help='After this step, timeout')
    parser.add_argument('--reward_clip',action='store_true',        help='Reward will be clipped in [-1, 1]')
    parser.add_argument('--render',     action='store_true',        help='First agent render')
    parser.add_argument('--load_model', action='store_true',        help='Load model in ./save_model/')
    parser.add_argument('--verbose',    action='store_true',        help='Print summary of model of global network')
    parser.add_argument('--debug',      action='store_true')
    args = parser.parse_args()
    
    env = Env()
    global_agent = PPOAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        seq_size=args.seqsize,
        gamma=args.gamma,
        lambd=args.lambd,
        entropy=args.entropy,
        horizon=args.horizon,
        actor_lr=args.lr,
        critic_lr=args.lr,
        batch_size=args.batch_size,
        epoch=args.epoch,
        clip=args.clip,
        thread_num=args.threads,
        load_model=args.load_model,
        verbose=args.verbose,
        render=args.render,
        save_rate=args.save_rate,
        reward_clip=args.reward_clip,
        debug=args.debug,
        timeout=args.timeout
    )
    global_agent.train()