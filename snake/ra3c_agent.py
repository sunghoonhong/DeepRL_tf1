'''
Author: Sunghoon Hong
Title: ra3c_agent.py
Version: 0.1.0
Description: RA3C Agent
Detail:
    Sequence size = 2
    RESIZE = 40
    Action size = 4
    Weight of Entropy of actor loss = 0.1
    Loss function of Critic = Huber loss
    lr = 2.5e-4
    20-step TD
    2-layer CNN
    LSTM output = 512
    Modify discounted_prediction()

'''

import os   
import time
import random
import threading
import csv
from datetime import datetime as dt
from collections import deque
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Input, LSTM, TimeDistributed, Reshape
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

SAVE_STAT_EPISODE_RATE = 100
SAVE_STAT_TIME_RATE = 600 #sec
THREAD_NUM = 32
RESIZE = 40
TRAIN = True
# TRAIN = False
LOAD_MODEL = True
VERBOSE = False

# Hyper Parameter
K_STEP = 20
ENT_WEIGHT = 0.1
LR = 2.5e-4

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
        self.discount = 0.95
        self.actor_lr = LR
        self.critic_lr = LR
        self.threads = THREAD_NUM
        self.lam = 0.6

        self.state_size = (self.seq_size, RESIZE, RESIZE)
        self.action_size = 4

        self.actor, self.critic = self.build_model()
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]
        
        self.stats = []

        if load_model:
            self.load_model('./save_model/ra3c')

    def train(self):
        agents = [Agent(tid, self.action_size, self.state_size,
                        [self.actor, self.critic], self.optimizer,
                        self.lam, self.discount, self.seq_size, self.build_model,
                        self.stats)
                  for tid in range(self.threads)]

        for agent in agents:
            time.sleep(1)
            agent.start()
        print(dt.now().strftime('%Y-%m-%d %H:%M:%S'), '%s agents Ready!' % self.threads)
        while True:
            # for i in range(30):
            #     print('Next Update After %d (sec)' % (300-i*10), end='\r', flush=True)
            #     time.sleep(10)
            time.sleep(SAVE_STAT_TIME_RATE)
            if self.stats:
                stats = np.copy(self.stats)
                self.stats.clear()
                with open('ra3c_output.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    for row in stats:
                        wr.writerow(row)
                self.save_model('./save_model/ra3c')
                mean = np.mean(np.float32(np.split(stats, [-1], axis=1)[0]), axis=0)
                print('%s\t%s Episodes Trained! AvgScore:%s AvgStep:%s AvgPmax:%s' 
                        % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), 
                        len(stats), mean[3], mean[1], mean[4]), end='/r')
                stats = None
            else:
                print('%s\tNo Episodes...' % (dt.now().strftime('%Y-%m-%d %H:%M:%S')))        

    def build_model(self):
        state_size = list(self.state_size)
        state_size.append(1)
        input = Input(shape=self.state_size)
        reshape = Reshape(state_size)(input)

        conv = TimeDistributed(Conv2D(16, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='he_normal'))(reshape)
        conv = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'))(conv)
        conv = TimeDistributed(Flatten())(conv)
        lstm = LSTM(512, activation='tanh', kernel_initializer='he_normal')(conv)

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

    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        loss = cross_entropy + ENT_WEIGHT * entropy

        optimizer = Adam(lr=self.actor_lr, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [],loss)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # Huber loss
        error = K.abs(discounted_prediction - value)
        quadratic = K.clip(error, 0.0, 1.0)
        linear = error - quadratic
        loss = K.mean(0.5 * K.square(quadratic) + linear)

        optimizer = Adam(lr=self.critic_lr, epsilon=0.01)
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

    def get_action(self, history):
        policy = self.actor.predict(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    def play(self, episodes=10, delay=0, improve='policy', debug=False):
        env = Env()
        scores = 0
        steps = 0
        print('Random\tGreedy\tPolicy')
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
                if debug:
                    for snap in history[0]:
                        Image.fromarray(snap*255.).show()
                time.sleep(delay)
                step += 1
                if self.render:
                    env.render()
                action, policy = self.get_action(history)
                if improve == 'greedy':
                    real_action = int(np.argmax(policy)) + 1
                elif improve == 'e-greedy':
                    real_action = int(np.argmax(policy)) + 1 if np.random.uniform(low=0.0, high=1.0) > 0.1 else action + 1
                else:
                    real_action = action + 1
                print(ACTION[action], '\t', ACTION[int(np.argmax(policy))], '\t', policy)
                if debug:
                    while True:
                        a = input('Press y or action(1(up),2(down),3(left),4(right)):')
                        if a==y:
                            break
                        elif a in ['1', '2', '3', '4']:
                            real_action = int(a)
                next_observe, reward, done, info = env.step(real_action)
                next_state = preprocess(next_observe)
                next_state = np.float32(next_state / 255.)
                next_state = np.reshape([next_state], (1, 1, RESIZE, RESIZE))
                next_history = np.append(next_state, history[:, :(self.seq_size-1), :, :], axis=1)

                history = next_history

            steps += step
            scores += env.game.score
            step = 0

            print('Score: %s Step: %s' % (env.game.score, step))
        return scores/episodes, steps/episodes


class Agent(threading.Thread):
    def __init__(self, tid, action_size, state_size, model,
                 optimizer, lam, discount, seq_size,
                 build_model, stats):
        threading.Thread.__init__(self)

        self.tid = tid
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.optimizer = optimizer
        self.discount = discount
        self.seq_size = seq_size
        self.stats = stats
        self.lam = lam

        self.states, self.actions, self.rewards = [], [], []

        self.local_actor, self.local_critic = build_model()
        self.update_local_model()

        self.reward_sum = 0
        self.avg_p_max = 0
        self.actor_loss = 0
        self.critic_loss = 0

        self.t_max = K_STEP
        self.t = 0
        # print('Agent %d Ready!' % self.tid)

    def run(self):
        global episode
        env = Env()

        step = 0

        while True:
            # print(self.tid, 'Still Training!')
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
                step += 1
                self.t += 1
                action, policy = self.get_action(history)
                next_observe, reward, done, info = env.step(action + 1)
                next_state = preprocess(next_observe)
                next_state = np.reshape([next_state], (1, 1, RESIZE, RESIZE))
                next_state = np.float32(next_state / 255.)
                next_history = np.append(next_state, history[:, :(self.seq_size-1), :, :], axis=1)

                self.avg_p_max += np.amax(policy)
                self.reward_sum += reward
                self.append_sample(history, action, reward)

                history = next_history

                if self.t >= self.t_max or done:
                    actor_loss, critic_loss = self.train_model(next_history, done)
                    self.actor_loss += abs(actor_loss[0])
                    self.critic_loss += abs(critic_loss[0])
                    self.update_local_model()
                    self.t = 0

                if done:
                    episode += 1
                    # print("episode:", episode, "  score:", env.game.score, "  step:", step)
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
                    self.avg_p_max = 0
                    self.reward_sum = 0
                    self.actor_loss = 0
                    self.critic_loss = 0
                    step = 0

    # def discounted_prediction(self, next_history, rewards, done):
    #     reward_pred = np.zeros_like(rewards)
    #     lambda_pred = np.zeros_like(rewards)
    #     R = R_lambda = 0
    #     value_pred = self.local_critic.predict(self.states)
    #     if not done:
    #         R = R_lambda = self.local_critic.predict(next_history)[0]

    #     for t in reversed(range(0, len(rewards))):
    #         R = self.discount * R + rewards[t]
    #         R_lambda = ( rewards[t] + self.discount * 
    #             (self.lam * R + (1-self.lam) * self.local_critic.predict(states[t+1]))
    #         )
    #         reward_pred[t] = R
    #     return reward_pred

    def train_model(self, next_history, done):
        # discounted_prediction = self.discounted_prediction(next_history, self.rewards, done)
        states = np.zeros((len(self.states) + 1, self.seq_size, RESIZE, RESIZE))
        for i in range(len(self.states)):
            states[i] = self.states[i]
        states[len(self.states)] = next_history
        values = self.critic.predict(states)
        values = np.reshape(values, len(values))

        # discounted prediction with TD lambda
        reward_pred = np.zeros_like(self.rewards)
        lambda_pred = np.zeros_like(self.rewards)
        R = R_lambda = 0
        if not done:
            R = R_lambda = values[-1]

        for t in reversed(range(0, len(self.rewards))):
            R = self.discount * R + self.rewards[t]
            R_lambda = ( self.rewards[t] + self.discount * 
                (self.lam * R_lambda + (1-self.lam) * values[t+1]) 
            )
            reward_pred[t] = R
            lambda_pred[t] = R_lambda

        adv_actor = reward_pred - values[:-1]
        adv_critic = lambda_pred - values[:-1]

        actor_loss = self.optimizer[0]([states[:-1], self.actions, adv_actor])
        critic_loss = self.optimizer[1]([states[:-1], adv_critic])
        self.states, self.actions, self.rewards = [], [], []
        return actor_loss, critic_loss

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
