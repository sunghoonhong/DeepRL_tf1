'''
Author: Sunghoon Hong
Title: addpg_agent.py
Version: 0.0.1
Description: Asynchronus Deep Deterministic Policy Gradient
Detail:
    10-step TD
'''


import os
import time
import csv
import random
import threading
import numpy as np
import gym
from keras import backend as K
from keras.layers import Dense, Conv2D, Flatten, Input, Lambda
from keras.layers import Reshape, Concatenate, LSTM, TimeDistributed
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
from PIL import ImageOps, Image
from snake2_env import Env


SAVE_EPISODE_RATE = 200
SAVE_STAT_TIME_RATE = 600

RESIZE = 40
SEQ_SIZE = 4
K_STEP = 10
THREAD_NUM = 32

LOAD_MODEL = True
VERBOSE = False
RENDER = True
RENDER = False
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


def preprocess(observe):
    ret = Image.fromarray(observe)
    ret = ImageOps.mirror(ret.rotate(270)).convert('L').resize((RESIZE, RESIZE))
    return np.asarray(ret)


class ADDPGAgent:
    def __init__(self, action_space, verbose=False, render=False, load_model=True):
        self.render = render
        self.verbose = verbose
        self.seq_size = SEQ_SIZE
        self.state_size = (self.seq_size, RESIZE, RESIZE)
        self.action_size = action_space.shape[0]
        self.action_low = action_space.low
        self.action_high = action_space.high
        
        # Hyper-Parameter
        self.threads = THREAD_NUM
        self.actor_lr = 1e-6
        self.critic_lr = 2.5e-5
        self.gamma = 0.99

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
            self.load_model('ddpg')

    def train(self):
        workers = [Worker(tid, self.action_size, [self.action_low, self.action_high], self.state_size,
                        [self.actor, self.critic], [self.actor_update, self.critic_update],
                        self.gamma, self.seq_size, self.build_model,
                        self.sess, self.stats)
                  for tid in range(self.threads)]

        for worker in workers:
            time.sleep(1)
            worker.start()

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
                stats = None
                print('Global Model Updated!')
    
    def play(self, episodes=3, delay=0.1, imporve='policy', debug=False):
        env = Env()
        for episode in range(1, episodes+1):
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
                if self.render:
                    env.render()
                step += 1
                action = self.get_action(history)
                if imporve=='exploration':
                    noise = np.random.normal()
                    action = np.clip(action + noise, self.action_low, self.action_high)
                print(action[0], action[0]*180/np.pi)
                if debug:
                    while True:
                        a = input('Press y or action[-pi, pi]:')
                        if a == 'y':
                            break
                        try:
                            a = float(a)
                            action = [a]
                        except:
                            continue
                next_observe, reward, done, info = env.step(action[0])
                next_state = preprocess(next_observe)
                next_state = np.reshape([next_state], (1, 1, RESIZE, RESIZE))
                next_state = np.float32(next_state / 255.)
                next_history = np.append(next_state, history[:, :(self.seq_size-1), :, :], axis=1)
                # self.reward_sum += reward

                history = next_history

                if done:
                    print("episode:", episode, "  score:", env.game.score, "  step:", step)

    def build_model(self):
        state_size = list(self.state_size)
        state_size.append(1)
        state = Input(shape=self.state_size)
        reshape = Reshape(state_size)(state)

        conv = TimeDistributed(Conv2D(16, (4, 4), strides=(2, 2), activation='relu'))(reshape)
        conv = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(conv)
        conv = TimeDistributed(Flatten())(conv)
        
        lstm_state = LSTM(256, activation='relu')(conv)
        action_output = Dense(self.action_size, activation='tanh')(lstm_state)
        actor_output = Lambda(lambda x: x * np.pi)(action_output)

        actor = Model(inputs=state, outputs=action_output)

        action = Input([self.action_size])
        state_action = Concatenate()([lstm_state, action])
        fc = Dense(256, activation='relu')(state_action)
        Q_output = Dense(1)(fc)
    
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
        train = K.function([self.actor.input, self.critic.input[1]], [],
                            updates=[updates])
        return train
        
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
        print('Model Saved\n')

    def load_model(self, name):
        if os.path.exists('save_model/' + name + '_actor.h5'):
            self.actor.load_weights('save_model/' + name + '_actor.h5')
            print('Actor Loaded')
        if os.path.exists('save_model/' + name + '_critic.h5'):
            self.critic.load_weights('save_model/' + name + '_critic.h5')
            print('Critic Loaded')


class Worker(threading.Thread):
    def __init__(self, tid, action_size, action_range, state_size,
                model, optimizer, gamma, seq_size,
                build_model, sess, stats):
        threading.Thread.__init__(self)

        self.tid = tid
        self.action_size = action_size
        self.action_low, self.action_high = action_range
        self.state_size = state_size
        self.actor, self.critic = model
        self.actor_update, self.critic_update = optimizer
        self.gamma = gamma
        self.seq_size = seq_size
        self.stats = stats

        self.states, self.actions, self.rewards = [], [], []

        self.local_actor, self.local_critic = build_model()
        self.update_local_model()

        self.reward_sum = 0
        self.critic_loss = 0

        self.t_max = K_STEP
        self.t = 0
        print('Agent %d Ready!' % self.tid)

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
                action = self.get_action(history)
                # print(action)
                next_observe, reward, done, info = env.step(action[0])
                next_state = preprocess(next_observe)
                next_state = np.reshape([next_state], (1, 1, RESIZE, RESIZE))
                next_state = np.float32(next_state / 255.)
                next_history = np.append(next_state, history[:, :(self.seq_size-1), :, :], axis=1)

                self.reward_sum += reward
                self.append_sample(history, action, reward)

                history = next_history

                if self.t >= self.t_max or done:
                    critic_loss = self.train_model(next_history, done)
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
                        self.reward_sum, env.game.score, avg_critic_loss,
                        info
                    ]
                    self.stats.append(stats)
                    # self.avg_Q = 0
                    self.reward_sum = 0
                    self.critic_loss = 0
                    step = 0

    def discounted_prediction(self, next_history, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0
        next_action = self.actor.predict(next_history)
        if not done:
            running_add = self.critic.predict([next_history, next_action])[0]

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

    def train_model(self, next_history, done):
        discounted_prediction = self.discounted_prediction(next_history, self.rewards, done)
        states = np.zeros((len(self.states), self.seq_size, RESIZE, RESIZE))
        for i in range(len(self.states)):
            states[i] = self.states[i]

        pred_actions = self.local_actor.predict(states)

        critic_loss = self.critic_update([states, self.actions, discounted_prediction])
        self.actor_update([states, pred_actions])

        self.states, self.actions, self.rewards = [], [], []
        return critic_loss

    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def get_action(self, history):
        [act] = self.actor.predict(history)
        noise = np.random.normal()
        act = np.clip(act + noise, self.action_low, self.action_high)
        return act

    def append_sample(self, history, action, reward):
        self.states.append(history)
        self.actions.append(action)
        self.rewards.append(reward)


if __name__ == '__main__':
    env = Env()
    agent = ADDPGAgent(env.action_space, verbose=VERBOSE, render=RENDER, load_model=LOAD_MODEL)
    if not os.path.exists('save_model'):
        os.makedirs('save_model')
    if not os.path.exists('save_graph'):
        os.makedirs('save_graph')
    agent.train()