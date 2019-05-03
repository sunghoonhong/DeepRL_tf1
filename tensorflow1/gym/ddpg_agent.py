'''
Author: Sunghoon Hong
Title: ddpg_agent.py
Description:
    Deep Deterministic Policy Gradient Agent for gym
Detail:

'''


import os
import csv
import time
import random
import argparse
from copy import deepcopy
from collections import deque
from datetime import datetime as dt
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import TimeDistributed, BatchNormalization, Flatten, Add, Lambda, Concatenate
from keras.layers import Dense, Input, ELU, Activation
from keras.optimizers import Adam
from keras.models import Model

import gym

np.set_printoptions(suppress=True, precision=4)
agent_name = 'ddpg'


class DDPGAgent(object):
    
    def __init__(self, state_size, action_size, actor_lr, critic_lr, tau,
                gamma, lambd, batch_size, memory_size,
                epsilon, epsilon_end, decay_step, load_model, game):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        self.lambd = lambd
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.decay_step = decay_step
        self.epsilon_decay = (epsilon - epsilon_end) / decay_step

        self.game = game
        self.sess = tf.Session()
        K.set_session(self.sess)

        self.actor, self.critic = self.build_model()
        self.target_actor, self.target_critic = self.build_model()
        self.actor_update = self.build_actor_optimizer()
        self.critic_update = self.build_critic_optimizer()
        self.sess.run(tf.global_variables_initializer())

        if load_model:
            self.load_model('./save_model/%s_%s' % (agent_name, game))
        
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.memory = deque(maxlen=self.memory_size)

    def build_model(self):
        # shared network
        state = Input(shape=[self.state_size])
        state_process = Dense(32, kernel_initializer='he_normal', use_bias=False)(state)
        state_process = BatchNormalization()(state_process)
        state_process = Activation('elu')(state_process)

        # Actor
        policy = Dense(64, kernel_initializer='he_normal', use_bias=False)(state_process)
        policy = BatchNormalization()(policy)
        policy = ELU()(policy)
        policy = Dense(self.action_size, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(policy)
        actor = Model(inputs=state, outputs=policy)
        
        # Critic
        action = Input(shape=[self.action_size])
        action_process = Dense(32, kernel_initializer='he_normal', use_bias=False)(action)
        action_process = BatchNormalization()(action_process)
        action_process = ELU()(action_process)
        state_action = Add()([state_process, action_process])

        Qvalue = Dense(64, kernel_initializer='he_normal', use_bias=False)(state_action)
        Qvalue = BatchNormalization()(Qvalue)
        Qvalue = ELU()(Qvalue)
        Qvalue = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(Qvalue)
        critic = Model(inputs=[state, action], outputs=Qvalue)

        actor._make_predict_function()
        critic._make_predict_function()
        
        return actor, critic

    def build_actor_optimizer(self):
        pred_Q = self.critic.output
        action_grad = tf.gradients(pred_Q, self.critic.input[1])
        target = -action_grad[0]
        loss = K.mean(target * self.actor.output)
        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, self.critic.input[1]], [loss],
                            updates=updates)#[updates])
        return train

    def build_critic_optimizer(self):
        y = K.placeholder(shape=(None, ), dtype='float32')
        pred = self.critic.output
        
        loss = 0.5 * K.mean(K.square(pred - y))
        # Huber Loss
        # error = K.abs(y - pred)
        # quadratic = K.clip(error, 0.0, 1.0)
        # linear = error - quadratic
        # loss = K.mean(0.5 * K.square(quadratic) + linear)

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input[0], self.critic.input[1], y],
                            [loss], updates=updates)
        return train

    def get_action(self, state):
        policy = self.actor.predict(state)[0]
        noise = [np.random.normal(scale=self.epsilon)] * self.action_size
        noise = np.array(noise, dtype=np.float64)
        action = policy + noise
        return action, policy
    
    def train_model(self):
        batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        actions = np.zeros((self.batch_size, self.action_size))
        rewards = np.zeros((self.batch_size, 1))
        next_states = np.zeros((self.batch_size, self.state_size))
        dones = np.zeros((self.batch_size, 1))

        targets = np.zeros((self.batch_size, 1))
        
        for i, sample in enumerate(batch):
            states[i] = sample[0]
            actions[i] = sample[1]
            rewards[i] = sample[2]
            next_states[i] = sample[3]
            dones[i] = sample[4]
        
        policy = self.actor.predict(states)
        target_actions = self.target_actor.predict(next_states)
        target_next_Qs = self.target_critic.predict([next_states, target_actions])
        targets = rewards + self.gamma * (1 - dones) * target_next_Qs

        actor_loss = self.actor_update([states, policy])
        critic_loss = self.critic_update([states, actions, targets])
        return actor_loss[0], critic_loss[0]

    def append_memory(self, state, action, reward, next_state, done):        
        self.memory.append((state, action, reward, next_state, done))
        
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

    def update_target_model(self):
        actor_weights = np.array(self.actor.get_weights())
        target_actor_weights = np.array(self.target_actor.get_weights())
        critic_weights = np.array(self.critic.get_weights())
        target_critic_weights = np.array(self.target_critic.get_weights())
        new_actor_weights = self.tau * actor_weights + (1 - self.tau) * target_actor_weights
        new_critic_weights = self.tau * critic_weights + (1 - self.tau) * target_critic_weights
        self.target_actor.set_weights(new_actor_weights)
        self.target_critic.set_weights(new_critic_weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',    action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--render',     action='store_true')
    parser.add_argument('--play',       action='store_true')
    parser.add_argument('--actor_lr',   type=float, default=1e-3)
    parser.add_argument('--critic_lr',  type=float, default=2.5e-3)
    parser.add_argument('--tau',        type=float, default=0.1)
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--lambd',      type=float, default=0.90)
    parser.add_argument('--batch_size', type=int,   default=512)
    parser.add_argument('--memory_size',type=int,   default=100000)
    parser.add_argument('--train_start',type=int,   default=5000)
    parser.add_argument('--train_rate', type=int,   default=100)
    parser.add_argument('--epsilon',    type=float, default=1.5)
    parser.add_argument('--epsilon_end',type=float, default=0.01)
    parser.add_argument('--decay_step', type=int,   default=200000)
    parser.add_argument('--game',       type=str,   default='MountainCarContinuous-v0')

    args = parser.parse_args()

    if not os.path.exists('save_graph/'+ agent_name):
        os.makedirs('save_graph/'+ agent_name)
    if not os.path.exists('save_stat'):
        os.makedirs('save_stat')
    if not os.path.exists('save_model'):
        os.makedirs('save_model')

    env = gym.make(args.game)
    # Make RL agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_high = env.action_space.high[0]

    agent = DDPGAgent(
        state_size=state_size,
        action_size=action_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        gamma=args.gamma,
        lambd=args.lambd,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        decay_step=args.decay_step,
        load_model=args.load_model,
        game=args.game
    )

    # Train
    episode = 0
    
    if args.play:
        while True:
            done = False

            # stats
            timestep, score, avgAct, avgQ = 0., 0, 0., 0.

            observe = env.reset()
            state = observe.reshape(1, -1)
            while not done:
                if args.render:
                    env.render()
                timestep += 1
                action = agent.actor.predict(state)[0]
                observe, reward, done, _ = env.step(action * action_high)
                next_state = observe.reshape(1, -1)

                # stats
                avgAct += float(action * action_high)
                avgQ += float(agent.critic.predict([state, action.reshape(1, -1)])[0][0])
                score += reward

                if timestep % 10 == 0:
                    print('%s' % (action * action_high), end='\r', flush=True)

                if args.verbose:
                    print('Step %d Action %s Reward %.2f' % (timestep, action, reward))

                state = next_state

                if agent.epsilon > agent.epsilon_end:
                    agent.epsilon -= agent.epsilon_decay

            # done
            avgAct /= timestep
            avgQ /= timestep

            print('Ep %d: Step %d Score %.2f AvgQ:%.3f AvgAct:%.3f' % (episode, timestep, score, avgQ, avgAct))

            episode += 1
    else:
        if os.path.exists('save_stat/%s_%s.csv' % (agent_name, args.game)):
            with open('save_stat/%s_%s.csv' % (agent_name, args.game), 'r') as f:
                read = csv.reader(f)
                episode = int(float(next(reversed(list(read)))[0]))
                print('Last episode:', episode)
                episode += 1
        stats = []
        while True:
            done = False

            # stats
            timestep, score, avgAct, avgQ = 0., 0, 0., 0.
            actor_loss, critic_loss = 0., 0.

            observe = env.reset()
            state = observe.reshape(1, -1)
            while not done:
                if args.render and episode % 5 == 0:
                    env.render()
                if len(agent.memory) >= args.train_start and timestep % args.train_rate == 0:
                    a_loss, c_loss = agent.train_model()
                    actor_loss += float(a_loss)
                    critic_loss += float(c_loss)
                    agent.update_target_model()
                timestep += 1
                action, policy = agent.get_action(state)
                observe, reward, done, _ = env.step(action * action_high)
                next_state = observe.reshape(1, -1)

                agent.append_memory(state, action, reward, next_state, done)

                # stats
                avgQ += float(agent.critic.predict([state, action.reshape(1, -1)])[0][0])
                avgAct += float(action * action_high)
                score += reward

                if timestep % 10 == 0:
                    print('%s | %s' % (action * action_high, policy * action_high), end='\r', flush=True)

                if args.verbose:
                    print('Step %d Action %s Reward %.2f' % (timestep, action, reward))

                state = next_state

                if agent.epsilon > agent.epsilon_end:
                    agent.epsilon -= agent.epsilon_decay

            # done
            actor_loss /= timestep
            critic_loss /= timestep
            avgQ /= timestep
            avgAct /= timestep

            print('Ep %d: Step %d Score %.2f AvgQ:%.3f AvgAct:%.3f' % (episode, timestep, score, avgQ, avgAct))
            
            # log stats
            with open('save_stat/%s_%s.csv' % (agent_name, args.game), 'a', encoding='utf-8', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(['%.4f' % s if type(s) is float else s for s in [episode, timestep, score, actor_loss, critic_loss, avgQ, avgAct]])
            agent.save_model('./save_model/%s_%s' % (agent_name, args.game))
            episode += 1    