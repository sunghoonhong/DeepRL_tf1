import time
import os
import csv
import random
from collections import deque
import numpy as np
import gym
from keras import backend as K
from keras.layers import Dense, Conv2D, Input, Reshape, Concatenate
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf

VERBOSE = False
# VERBOSE = True
RENDER = True
# RENDER = False
# TRAIN = False
# TRAIN = True

class DDPGAgent:
    def __init__(self, state_space, action_space, render=False):
        self.state_size = state_space.shape[0]
        self.action_size = action_space.shape[0]
        self.action_low = action_space.low
        self.action_high = action_space.high
        
        # Hyper-Parameter
        self.lr = 2.5e-5
        self.gamma = 0.99
        self.memory_size = 1000000
        self.train_start = 10
        self.update_target_rate = 10
        self.batch_size = 32
        self.good_batch_size = 4
        self.normal_size = self.batch_size - self.good_batch_size

        # TF Session
        self.sess = tf.Session()
        K.set_session(self.sess)

        # Model
        self.actor, self.critic = self.build_model()
        self.target_actor, self.target_critic = self.build_model()
        self.update_target_model()
        self.action_grad = tf.gradients(self.critic.output, self.critic.input[1])
        self.actor_optimize = self.actor_optimizer()
        # self.get_action_grad = self.get_action_gradients()
        self.critic_update = self.critic_optimizer()

        self.sess.run(tf.global_variables_initializer())

        # Replay Memory
        self.memory = deque(maxlen=self.memory_size)
        self.good_memory = deque(maxlen=self.memory_size)

    def build_model(self):
        state = Input([self.state_size], batch_shape=[self.batch_size, self.state_size])
        fc1 = Dense(256, activation='relu', kernel_initializer='he_normal')(state)
        action_output = Dense(self.action_size, activation='linear', kernel_initializer='he_normal')(fc1)

        actor = Model(inputs=state, outputs=action_output)

        # state = Input([self.state_size], batch_shape=[self.batch_size, self.state_size])
        action = Input([self.action_size], batch_shape=[self.batch_size, self.action_size])
        state_action = Concatenate()([state, action])
        fc2 = Dense(256, activation='relu', kernel_initializer='he_normal')(state_action)
        Q_output = Dense(1, activation='linear', kernel_initializer='he_normal')(fc2)
    
        critic = Model(inputs=[state, action], outputs=Q_output)
        # action_grad = tf.gradients(critic.output, action)

        actor._make_predict_function()
        critic._make_predict_function()
        
        if VERBOSE:
            actor.summary()
            critic.summary()

        return actor, critic
    
    def actor_optimizer(self):
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_size])
        params_grad = tf.gradients(
            self.actor.output, self.actor.trainable_weights, -self.action_gradient)
        grads = zip(params_grad, self.actor.trainable_weights)
        optimizer = tf.train.AdamOptimizer(self.lr)
        updates = optimizer.apply_gradients(grads)
        return updates
        
    def actor_update(self, states, action_grad):
        return self.sess.run(self.actor_optimize, feed_dict={
            self.actor.input: states,
            self.action_gradient: action_grad
        })

    def get_action_gradients(self, states, actions):
        return self.sess.run(self.action_grad, feed_dict={
            self.actor.input: states,
            self.critic.input[1]: actions
        })[0]

    def critic_optimizer(self):
        y = K.placeholder(shape=(None, ), dtype='float32')
        pred = self.critic.output

        # Huber Loss
        error = K.abs(y-pred)
        quadratic = K.clip(error, 0.0, 1.0)
        linear = error - quadratic
        loss = K.mean(0.5 * K.square(quadratic) + linear)
        optimizer = Adam(lr=self.lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        # print(updates)
        train = K.function([self.critic.input[0], self.critic.input[1], y],
                            [loss], updates=updates)
        return train

    def update_target_model(self):
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def get_action(self, state):
        act = self.actor.predict(state)
        # if TRAIN:
        noise = np.random.normal()
        act = np.clip(act + noise, self.action_low, self.action_high)
        return act

    def train(self):
        batch = random.sample(self.memory, self.normal_size)
        good_batch = random.sample(self.good_memory, self.good_batch_size)

        states = np.zeros((self.batch_size, 2))
        actions = np.zeros((self.batch_size, 1))
        rewards = np.zeros((self.batch_size, 1))
        next_states = np.zeros((self.batch_size, 2))
        dones = np.zeros((self.batch_size, 1))
        
        for i, sample in enumerate(batch + good_batch):

            states[i] = sample[0]   #shape = (1, 2)
            actions[i] = sample[1]  #shape = (1, 1)
            rewards[i] = sample[2]  #shape = (,)
            next_states[i] = sample[3]  #shape = (1, 2)
            dones[i] = sample[4]    #shape = (,)
        # print('states',  states.shape, states)
        # print('actions', actions.shape, actions)
        # print('rewards', rewards.shape, rewards)
        # print('dones',  dones.shape ,dones)
        pred_action = self.actor.predict(states)
        target_action = self.target_actor.predict(next_states)
        target_Q = self.target_critic.predict([next_states, target_action])
        target_value = rewards + (1 - dones) * self.gamma * target_Q

        self.critic_update([states, actions, target_value])
        action_grad = self.get_action_gradients(states, pred_action)
        self.actor_update(states, action_grad)

    def append_sample(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))
        if r > 0:
            self.good_memory.append((s, a, r, ns, d))

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

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    agent = DDPGAgent(env.observation_space, env.action_space, RENDER)

    while True:
        done = False
        step = 0
        score = 0
        state = env.reset()
        print(state)
        state = np.reshape(state, [1, agent.state_size])
        while not done:
            time.sleep(5)
            if RENDER:
                env.render()
            print('State:', state)
            action = agent.get_action(state)
            print('Action:', action)
            next_state, reward, done, _ = env.step(action)
            print('Reward:', reward)
            step += 1
            score += reward
            next_state = np.reshape(next_state, [1, agent.state_size])

            # if TRAIN:
            agent.append_sample(state, action, reward, next_state, done)

            state = next_state
            # if TRAIN:
            #     if len(agent.good_memory) >= agent.good_batch_size:
            #         agent.train()
            #     if e % agent.update_target_rate == 1:
            #         agent.update_target_model()
            if done:
                e += 1
                # if TRAIN:
                #     stat = [e, step, score]
                #     stats.append(stat)
                #     with open('output.csv', 'a', encoding='utf-8', newline='') as f:
                #         wr = csv.writer(f)
                #         wr.writerow(stat)
                #     if e % 20 == 0 and len(agent.good_memory) >= agent.good_batch_size:
                #         avgScore = np.mean(stats, axis=0)[2]
                #         if avgScore > bestScore:
                #             bestScore = avgScore
                #             print('E: %s Best AvgScore: %s' % (e, bestScore))
                #             agent.save_model('ddpg')
                #         stats.clear()

                # else:                    
                print('step: %s, \t\t\tscore: %s' % (step, score))
