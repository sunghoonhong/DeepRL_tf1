'''
Author: Sunghoon Hong
Title: addpg_agent.py
Version: 0.0.3
Description: Asynchronus Deep Deterministic Policy Gradient
Detail:
    80-step TD(lambda)
'''


import os
import time
import csv
import random
import threading
import argparse
from datetime import datetime as dt
import numpy as np
import gym
from keras import backend as K
from keras.layers import Dense, Conv2D, Flatten, Input, Lambda, BatchNormalization
from keras.layers import Reshape, Concatenate, GRU, TimeDistributed
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
from PIL import ImageOps, Image


parser = argparse.ArgumentParser()
parser.add_argument('--save_rate', type=int, default=300)
parser.add_argument('--threads', type=int, default=32)
parser.add_argument('--tmax', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lambd', type=float, default=0.7)
parser.add_argument('--gamma', type=float, default=0.99)

args = parser.parse_args()

SAVE_EPISODE_RATE = 200
SAVE_STAT_TIME_RATE = args.save_rate

K_STEP = args.tmax
THREAD_NUM = args.threads
LAMBDA = args.lambd
GAMMA = args.gamma
LR = args.lr

LOAD_MODEL = True
VERBOSE = False
RENDER = True
# RENDER = False
# TRAIN = False
TRAIN = True

global episode
episode = 0
if not os.path.exists('save_graph/addpg_agent'):
    os.makedirs('save_graph/addpg_agent')
if not os.path.exists('save_model'):
    os.makedirs('save_model')
if os.path.exists('addpg_output.csv'):
    with open('addpg_output.csv', 'r') as f:
        read = csv.reader(f)
        episode = int(float(next(reversed(list(read)))[0]))
    print(episode)


class ADDPGAgent:
    def __init__(self, state_space, action_space, verbose=False, render=False, load_model=True):
        self.render = render
        self.verbose = verbose
        self.state_size = state_space.shape[0]
        self.action_size = action_space.shape[0]
        self.action_low = action_space.low
        self.action_high = action_space.high
        
        # Hyper-Parameter
        self.threads = THREAD_NUM
        self.actor_lr = LR
        self.critic_lr = LR
        self.gamma = GAMMA
        self.lambd = LAMBDA

        # TF Session
        self.sess = tf.Session()
        K.set_session(self.sess)

        # Model
        self.actor, self.critic = self.build_model()
        self.actor_update = self.actor_optimizer()
        self.critic_update = self.critic_optimizer()

        self.sess.run(tf.global_variables_initializer())

        self.stats = []

        if VERBOSE:
            self.actor.summary()
            self.critic.summary()

        if load_model:
            self.load_model('addpg')

    def train(self):
        workers = [Worker(tid, self.action_size, [self.action_low, self.action_high], self.state_size,
                        [self.actor, self.critic], [self.actor_update, self.critic_update],
                        self.gamma, self.lambd, self.build_model,
                        self.sess, self.stats)
                  for tid in range(self.threads)]

        for worker in workers:
            time.sleep(1)
            worker.start()
        print(dt.now().strftime('%Y-%m-%d %H:%M:%S'), '%s Workers Ready!' % self.threads)
        while True:
            # for i in range(30):
            #     print('Next Update After %d (sec)' % (300-i*10), end='\r', flush=True)
            #     time.sleep(10)
            time.sleep(SAVE_STAT_TIME_RATE)
            if self.stats:
                stats = np.copy(self.stats)
                self.stats.clear()
                with open('addpg_output.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    for row in stats:
                        wr.writerow(row)
                self.save_model('addpg')
                mean = np.mean(stats, axis=0)
                print('%s: %s Episodes Trained! AvgReward:%s AvgStep:%s' 
                        % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), 
                        len(stats), mean[2], mean[1]))
                stats = None
            else:
                print('%s: No Episodes...' % (dt.now().strftime('%Y-%m-%d %H:%M:%S')))
                    
    def play(self, episodes=3, delay=0.1, imporve='policy', debug=False, SNAPSHOT=False):
        env = gym.make('MountainCarContinuous-v0')
        # for episode in range(1, episodes+1):
        #     step = 0
        #     done = False
        #     observe = env.reset()

        #     while not done:
        #         time.sleep(delay)
        #         if self.render:
        #             env.render()
        #         if SNAPSHOT:
        #             snapshot = np.array([]).reshape([0,RESIZE])
        #             for snap in history[0]:
        #                 snapshot = np.append(snapshot, snap, axis=0)
        #             Image.fromarray(snapshot*255.).show()
        #         step += 1
        #         action = self.get_action(history)
        #         if imporve=='exploration':
        #             noise = np.random.uniform(low=-1.0, high=1.0) * np.pi
        #             # noise = np.random.normal(scale=2)
        #             action = np.clip(action + noise, self.action_low, self.action_high)
        #         print(action[0], action[0]*180/np.pi)
        #         if debug:
        #             while True:
        #                 a = input('Press y or action[-pi, pi]:')
        #                 if a == 'y':
        #                     break
        #                 try:
        #                     a = float(a)
        #                     action = [a]
        #                     break
        #                 except:
        #                     continue
        #         next_observe, reward, done, info = env.step(action[0])
        #         next_state = preprocess(next_observe).reshape((1, RESIZE, RESIZE))
        #         next_state = np.float32(next_state / 255.)
        #         next_history = np.append(history[0][1:], next_state, axis=0)
        #         next_history = np.float32([next_history])
        #         history = next_history

        #         history = next_history

                # if done:
                #     print("episode:", episode, "  score:", env.game.score, "  step:", step)

    def build_model(self):
        state = Input(shape=[self.state_size])
        fc1 = Dense(256, activation='elu', kernel_initializer='he_normal')(state)

        action_output = Dense(self.action_size, activation='tanh', kernel_initializer='he_normal')(state)

        actor = Model(inputs=state, outputs=action_output)

        action = Input([self.action_size])
        state_action = Concatenate()([state, action])
        fc = Dense(256, activation='elu', kernel_initializer='he_normal')(state_action)
        Q_output = Dense(1, kernel_initializer='he_normal')(fc)
    
        critic = Model(inputs=[state, action], outputs=Q_output)


        actor._make_predict_function()
        critic._make_predict_function()        

        return actor, critic
    
    def actor_optimizer(self):
        action_grad = tf.gradients(self.critic.output, self.critic.input[1])
        target = tf.math.negative(action_grad)
        # target = tf.reshape(target, shape=target.shape[1:])
        params_grad = tf.gradients(
            self.actor.output, self.actor.trainable_weights, target)
        grads = zip(params_grad, self.actor.trainable_weights)
        optimizer = tf.train.AdamOptimizer(self.actor_lr)
        updates = optimizer.apply_gradients(grads)
        # print('actor updates', updates)
        # train = K.function([self.actor.input, self.critic.input[1]], [],
                            # updates=[])
        return updates
        
    def critic_optimizer(self):
        y = K.placeholder(shape=(None, ), dtype='float32')
        pred = self.critic.output

        # Huber Loss
        error = K.abs(y-pred)
        quadratic = K.clip(error, 0.0, 1.0)
        linear = error - quadratic
        loss = K.mean(0.5 * K.square(quadratic) + linear)
        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        # print('critic update', updates)
        train = K.function([self.critic.input[0], self.critic.input[1], y],
                            [loss], updates=updates)
        return train

    def get_action(self, history):
        [act] = self.actor.predict(history)
        # noise = np.random.normal()
        # act = np.clip(act + noise, self.action_low, self.action_high)
        return act

    def save_model(self, name):
        self.actor.save_weights('save_model/' + name + '_actor.h5')
        self.critic.save_weights('save_model/' + name + '_critic.h5')

    def load_model(self, name):
        if os.path.exists('save_model/' + name + '_actor.h5'):
            self.actor.load_weights('save_model/' + name + '_actor.h5')
            print('Actor Loaded')
        if os.path.exists('save_model/' + name + '_critic.h5'):
            self.critic.load_weights('save_model/' + name + '_critic.h5')
            print('Critic Loaded')


class Worker(threading.Thread):
    def __init__(self, tid, action_size, action_range, state_size,
                model, optimizer, gamma, lambd,
                build_model, sess, stats):
        threading.Thread.__init__(self)

        self.tid = tid
        self.action_size = action_size
        self.action_low, self.action_high = action_range
        self.state_size = state_size
        self.actor, self.critic = model
        self.actor_update, self.critic_update = optimizer
        self.gamma = gamma
        self.lambd = lambd
        self.stats = stats
        self.render = (RENDER and tid==0)
        self.sess = sess

        self.states, self.actions, self.rewards = [], [], []

        self.local_actor, self.local_critic = build_model()
        self.update_local_model()

        self.reward_sum = 0
        self.critic_loss = 0

        self.t_max = K_STEP
        self.t = 0
        # print('Agent %d Ready!' % self.tid)

    def run(self):
        global episode
        env = gym.make('MountainCarContinuous-v0')

        step = 0

        while True:
            # print(self.tid, 'Still Training!')
            done = False
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])

            while not done:
                step += 1
                self.t += 1
                if self.render:
                    env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                # if REWARD_CLIP == 'clip':
                    # reward = np.clip(reward, -1.0, 1.0)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.reward_sum += reward
                self.append_sample(state, action, reward)

                state = next_state

                if self.t >= self.t_max or done:
                    critic_loss = self.train_model(next_state, done)
                    self.critic_loss += abs(critic_loss[0])
                    self.update_local_model()
                    self.t = 0

                if done:
                    episode += 1
                    # print("episode:", episode, "  score:", env.game.score, "  step:", step)
                    train_num = step // self.t_max + 1
                    avg_critic_loss = self.critic_loss / train_num
                    stats = [
                        episode, step,
                        self.reward_sum, avg_critic_loss
                    ]
                    self.stats.append(stats)
                    # self.avg_Q = 0
                    self.reward_sum = 0
                    self.critic_loss = 0
                    step = 0

    def actor_train(self, states, actions):
        self.sess.run(self.actor_update, feed_dict={
            self.actor.input: states,
            self.critic.input[1]: actions
        })

    def train_model(self, next_history, done):
        if self.lambd == 1:
            states = np.zeros((len(self.states), self.state_size))
            for i in range(len(self.states)):
                states[i] = self.states[i]
            pred_action = self.actor.predict(next_history)
            Qvalue = self.critic.predict([next_history, pred_action])
            G = 0
            rewards = np.zeros_like(self.rewards)
            if not done:
                G = Qvalue
            for t in reversed(range(0, len(self.rewards))):
                G = self.rewards[t] + self.gamma * G
                rewards[t] = G
            rewards /= 10
        
            pred_actions = self.actor.predict(states)
            self.actor_train(states, pred_actions)
            critic_loss = self.critic_update([states, self.actions, rewards])

        else:
            states = np.zeros((len(self.states) + 1, self.state_size))
            for i in range(len(self.states)):
                states[i] = self.states[i]
            states[len(self.states)] = next_history

            pred_actions = self.actor.predict(states)

            Qvalues = self.critic.predict([states, pred_actions])
            Qvalues = np.reshape(Qvalues, len(Qvalues))

            # discounted prediction with TD lambda
            lambda_pred = np.zeros_like(self.rewards)
            G_lambda = 0
            if not done:
                G_lambda = Qvalues[-1]
            else:
                Qvalues[-1] = np.float32([0])
            for t in reversed(range(0, len(self.rewards))):
                G_lambda = ( self.rewards[t] + self.gamma * 
                    (self.lambd * G_lambda + (1-self.lambd) * Qvalues[t+1]) 
                )
                lambda_pred[t] = G_lambda

            adv_critic = lambda_pred - Qvalues[:-1]

            critic_loss = self.critic_update([states[:-1], self.actions, adv_critic])
            self.actor_train(states[:-1], pred_actions[:-1])
        
        self.states, self.actions, self.rewards = [], [], []
        return critic_loss

    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def get_action(self, state):
        [act] = self.actor.predict(state)
        # noise = np.random.uniform(low=-np.pi, high=np.pi)
        global episode
        if episode < 1000:
            noise = np.random.normal(0.3, 1.5) + 0.5
        else:
            noise = np.random.normal(0, 1)
        act = np.clip(act + noise, self.action_low, self.action_high)
        return act

    def append_sample(self, history, action, reward):
        self.states.append(history)
        self.actions.append(action)
        self.rewards.append(reward)


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    agent = ADDPGAgent(env.observation_space, env.action_space, verbose=VERBOSE, render=RENDER, load_model=LOAD_MODEL)
    if not os.path.exists('save_model'):
        os.makedirs('save_model')
    if not os.path.exists('save_graph'):
        os.makedirs('save_graph')
    agent.train()