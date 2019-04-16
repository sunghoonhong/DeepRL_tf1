'''
D4PG

'''
import os
import csv
import time
import random
import argparse
import threading
from collections import deque
import numpy as np
import gym
from keras import backend as K
from keras.layers import Dense, Conv2D, Input, Reshape, Concatenate, BatchNormalization, Add
from keras.optimizers import Adam
from keras.models import Model
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

global episode
episode = 0
if not os.path.exists('save_graph/d4pg_agent'):
    os.makedirs('save_graph/d4pg_agent')
if not os.path.exists('save_model'):
    os.makedirs('save_model')
if os.path.exists('d4pg_output.csv'):
    with open('d4pg_output.csv', 'r') as f:
        read = csv.reader(f)
        episode = int(float(next(reversed(list(read)))[0]))
    print(episode)

parser = argparse.ArgumentParser()
parser.add_argument('--tmax', type=int, default=1000)
parser.add_argument('--threads', type=int, default=1)
parser.add_argument('--render', action='store_true')
# parser.add_argument('--lambd', type=float, default=1.0)
args = parser.parse_args()
TMAX = args.tmax

ACTOR_RATE = 3


VERBOSE = False
# VERBOSE = True
RENDER = args.render
# RENDER = True
# TRAIN = False
TRAIN = True

class D4PGAgent:
    def __init__(self, state_space, action_space, render=False):
        self.state_size = state_space.shape[0]
        self.action_size = action_space.shape[0]
        self.action_low = action_space.low
        self.action_high = action_space.high
        
        # Hyper-Parameter
        self.tau = 0.1
        self.actor_lr = 1e-4
        self.critic_lr = 1e-4
        self.gamma = 0.99
        self.memory_size = 20000
        self.threads = args.threads
        # self.train_start = 1
        # self.update_target_rate = 10
        self.batch_size = 100

        # TF Session
        self.sess = tf.Session()
        K.set_session(self.sess)

        # Model
        self.actor, self.critic = self.build_model()
        self.target_actor, self.target_critic = self.build_model()
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_update = self.actor_optimizer()
        self.critic_update = self.critic_optimizer()

        self.sess.run(tf.global_variables_initializer())

        # Replay Memory
        self.memory = deque(maxlen=self.memory_size)


    def train(self):
        env = gym.make('MountainCarContinuous-v0')

        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        action_examples = np.array([env.action_space.sample() for x in range(10000)])
        scaler_s = StandardScaler()
        scaler_a = StandardScaler()
        scaler_s.fit(observation_examples)
        scaler_a.fit(action_examples)

        agents = [Agent(tid, self.action_size, self.state_size,
                        [self.actor, self.critic], [self.target_actor, self.target_critic],
                        [self.actor_update, self.critic_update], self.build_model,
                        self.gamma, self.memory, scaler_s, scaler_a)
                  for tid in range(self.threads)]
        for agent in agents:
            time.sleep(1)
            agent.start()

        t = 0
        while True:
            t += 1
            if len(self.memory) > self.batch_size:
                self.train_memory()
                self.update_target_model()
                if t == self.threads :
                    for agent in agents:
                        agent.update_local_model()
                    t = 0

    def train_memory(self):
        batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((0, 2))
        actions = np.zeros((0, 1))
        # rewards = np.zeros((0, 1))
        discounted = np.zeros((0, 1))
    
        for transition in batch:
            s = transition[0]
            a = transition[1]
            r = transition[2]
            done = transition[3]
            if done:
                G = 0
            else:
                # print(s.shape, s[-1].shape, np.shape([s[-1]]))
                pred_action = self.target_actor.predict(np.reshape(s[-1], (1,2)))
                pred_Q = self.target_critic.predict([np.reshape(s[-1], (1,2)), pred_action])
                G = pred_Q
            discount = np.zeros([len(r), 1])
            for t in reversed(range(len(r))):
                G = r[t] + self.gamma * G
                discount[t] = G
            # print(np.shape(s), np.shape(a), np.shape(r))
            states = np.append(states, s[:-1], axis=0)
            actions = np.append(actions, a, axis=0)
            # rewards = np.append(rewards, r, axis=0)
            discounted = np.append(discounted, discount, axis=0)
        pred_actions = self.target_actor.predict(states)
        # print(states.shape, actions.shape, pred_actions.shape, discounted.shape)
        self.actor_update([states, pred_actions])
        self.critic_update([states, actions, discounted])

    def build_model(self):
        state = Input([self.state_size])
        fc1 = Dense(100, activation='elu', kernel_initializer='he_normal')(state)
        action_output = Dense(self.action_size, activation='tanh', kernel_initializer='he_normal')(fc1)

        actor = Model(inputs=state, outputs=action_output)

        action = Input([self.action_size])
        action_process = Dense(100, activation='elu', kernel_initializer='he_normal')(action)
        state_action = Add()([fc1, action_process])
        fc2 = Dense(50, activation='elu', kernel_initializer='he_normal')(state_action)
        Q_output = Dense(1, kernel_initializer='he_normal')(fc2)
    
        critic = Model(inputs=[state, action], outputs=Q_output)

        actor._make_predict_function()
        critic._make_predict_function()
        
        if VERBOSE:
            actor.summary()
            critic.summary()

        return actor, critic
    
    def actor_optimizer(self):
        action_grad = tf.gradients(self.critic.output, self.critic.input[1])
        target = tf.math.negative(action_grad)

        params_grad = tf.gradients(
            self.actor.output, self.actor.trainable_weights, target)
        grads = zip(params_grad, self.actor.trainable_weights)
        optimizer = tf.train.AdamOptimizer(self.actor_lr)
        updates = optimizer.apply_gradients(grads)
        train = K.function([self.actor.input, self.critic.input[1]], [],
                            updates=[updates])
        return train

    def critic_optimizer(self):
        y = K.placeholder(shape=(None, 1), dtype='float32')
        pred = self.critic.output
        
        loss = K.mean(K.square(pred-y))
        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input[0], self.critic.input[1], y],
                            [loss], updates=updates)
        return train

    def update_target_model(self):
        copy_op = []
        tau = self.tau
        for main_var, target_var in zip(self.actor.trainable_weights, self.target_actor.trainable_weights):
            copy_op.append(target_var.assign(tf.multiply(main_var.value(), tau) + tf.multiply(target_var.value(), 1 - tau)))
        self.sess.run(copy_op)
        copy_op = []
        for main_var, target_var in zip(self.critic.trainable_weights, self.target_critic.trainable_weights):
            copy_op.append(target_var.assign(tf.multiply(main_var.value(), tau) + tf.multiply(target_var.value(), 1 - tau)))
        self.sess.run(copy_op)


class Agent(threading.Thread):
    def __init__(self, tid, action_size, state_size,
                model, target_model, optimizer, build_model, gamma, memory,
                scaler_s, scaler_a):
        threading.Thread.__init__(self)

        self.tid = tid
        self.action_size = action_size
        self.state_size = state_size
        self.action_low = -1.0
        self.action_high = 1.0
        self.actor, self.critic = model
        self.target_actor, self.target_critic = target_model
        self.actor_update, self.critic_update = optimizer
        self.gamma = gamma
        self.scaler_s = scaler_s
        self.scaler_a = scaler_a
        self.render = RENDER and tid==0

        # Model
        self.local_actor, self.local_critic = build_model()
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

        self.memory = memory

    def get_action(self, state):
        act = self.local_actor.predict(state)
        noise = np.random.normal(0, 1) * 0.3
        # noise = 0
        act = np.clip(act + noise, self.action_low, self.action_high)
        return act

    def run(self):
        env = gym.make('MountainCarContinuous-v0')
        
        global episode
        T_actor = 0

        while True:
            done = False
            step = 0
            T = 0
            # T_actor += 1
            # if T_actor >= ACTOR_RATE:
            #     self.update_local_model()
            #     T_actor = 0

            score = 0
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            state = list(self.scaler_s.transform(state)[0])
            state = np.reshape(state, [1, self.state_size])
            state_list, action_list, reward_list = [], [], []

            while not done:
                if self.render:
                    env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                T += 1
                step += 1
                next_state = np.reshape(next_state, [1, self.state_size])
                next_state = list(self.scaler_s.transform(next_state)[0])
                next_state = np.reshape(next_state, [1, self.state_size])
                action = self.scaler_a.transform(np.reshape(action, (-1, 1)))[0]
                score += reward
                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward)

                state = next_state

                if T >= TMAX or done:
                    T = 0
                    # next_action = self.target_actor.predict(next_state)
                    # Q = self.target_critic.predict([next_state, next_action])
                    # discounted = np.zeros_like(reward_list)
                    # G = 0 if done else Q

                    # for t in reversed(range(len(reward_list))):
                    #     G = reward_list[t] + self.gamma * G
                    #     discounted[t] = G
                    state_list.append(next_state)
                    self.append_sample(state_list, action_list, reward_list, done)
                    state_list, action_list, reward_list = [], [], []
                
            episode += 1
            print('Ep %s Score: %s Step: %s' % (episode, score, step))

    def append_sample(self, s, actions, rewards, done):
        states = np.zeros((len(s), 2))
        for i in range(len(s)):
            states[i] = s[i]
        self.memory.append((states, actions, rewards, done))

    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    agent = D4PGAgent(env.observation_space, env.action_space, RENDER)
    # agent.load_model('d4pg')
    # EPISODES = 500000
    if not os.path.exists('save_model'):
        os.makedirs('save_model')
    e = 0
    if os.path.exists('d4pg_output.csv'):
        with open('d4pg_output.csv', 'r') as f:
            read = csv.reader(f)
            e = int(next(reversed(list(read)))[0])
        print(e)
    np.set_printoptions(precision=4, suppress=True)

    agent.train()