'''
DDPG with ER

'''
import os
import csv
import random
import argparse
from collections import deque
import numpy as np
import gym
from keras import backend as K
from keras.layers import Dense, Conv2D, Input, Reshape, Concatenate, BatchNormalization, Add
from keras.optimizers import Adam
from keras.models import Model
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

VERBOSE = False
# VERBOSE = True
RENDER = False
# RENDER = True
# TRAIN = False
TRAIN = True

class DDPGAgent:
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
        # self.train_start = 1
        # self.update_target_rate = 10
        self.batch_size = 50

        # TF Session
        self.sess = tf.Session()
        K.set_session(self.sess)

        # Model
        self.actor, self.critic = self.build_model()
        self.target_actor, self.target_critic = self.build_model()
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        # self.get_action_grad = self.get_action_gradients()
        self.actor_update = self.actor_optimizer()
        self.critic_update = self.critic_optimizer()

        self.sess.run(tf.global_variables_initializer())

        # Replay Memory
        self.memory = deque(maxlen=self.memory_size)

    def build_model(self):
        state = Input([self.state_size])
        fc1 = Dense(100, activation='elu')(state)
        # fc1 = BatchNormalization()(fc1)
        # fc1 = Dense(100, activation='elu')(fc1)
        action_output = Dense(self.action_size, activation='tanh')(fc1)

        actor = Model(inputs=state, outputs=action_output)

        # state = Input([self.state_size], batch_shape=[self.batch_size, self.state_size])
        action = Input([self.action_size])
        action_process = Dense(100, activation='elu')(action)
        # action_process = BatchNormalization()(action_process)
        # state_process  = Dense(100, activation='elu')(state)
        # state_process  = BatchNormalization()(state_process)
        # state_process  = Dense(100, activation='elu')(state_process)
        state_action = Add()([fc1, action_process])
        fc2 = Dense(50, activation='elu')(state_action)
        Q_output = Dense(1)(fc2)
    
        critic = Model(inputs=[state, action], outputs=Q_output)
        # action_grad = tf.gradients(critic.output, action)

        actor._make_predict_function()
        critic._make_predict_function()
        
        if VERBOSE:
            actor.summary()
            critic.summary()

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

    # def actor_update(self, states, actions):
    #     self.sess.run(self.actor_optimize, feed_dict={
    #         self.actor.input: states,
    #         self.critic.input[1]: actions
    #     })
    # def actor_update(self, states, action_grad):
    #     return self.sess.run(self.actor_optimize, feed_dict={
    #         self.actor.input: states,
    #         self.action_gradient: action_grad
    #     })

    # def get_action_gradients(self, states, actions):
    #     return self.sess.run(self.action_grad, feed_dict={
    #         self.actor.input: states,
    #         self.critic.input[1]: actions
    #     })[0]

    def critic_optimizer(self):
        y = K.placeholder(shape=(None, ), dtype='float32')
        pred = self.critic.output
        
        loss = K.mean(K.square(pred-y))
        # Huber Loss
        # error = K.abs(y-pred)
        # quadratic = K.clip(error, 0.0, 1.0)
        # linear = error - quadratic
        # loss = K.mean(0.5 * K.square(quadratic) + linear)

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        # print(updates)
        train = K.function([self.critic.input[0], self.critic.input[1], y],
                            [loss], updates=updates)
        return train

    def update_target_model(self):
        # self.target_actor.set_weights(self.actor.get_weights())
        # self.target_critic.set_weights(self.critic.get_weights())
        copy_op = []
        tau = self.tau
        for main_var, target_var in zip(self.actor.trainable_weights, self.target_actor.trainable_weights):
            copy_op.append(target_var.assign(tf.multiply(main_var.value(), tau) + tf.multiply(target_var.value(), 1 - tau)))
        self.sess.run(copy_op)
        copy_op = []
        for main_var, target_var in zip(self.critic.trainable_weights, self.target_critic.trainable_weights):
            copy_op.append(target_var.assign(tf.multiply(main_var.value(), tau) + tf.multiply(target_var.value(), 1 - tau)))
        self.sess.run(copy_op)

    def get_action(self, state, ep):
        # if ep < 20:
        #     action = np.random.normal(0.3, 1.5) + 0.5
        #     action = np.clip(action, -1.0, 1.0)
        #     return [action]
        # # if TRAIN:
        # elif ep < 1000:
        #     ep /= 100
        #     c = 1 - ep/10
        act = self.actor.predict(state)
        noise = np.random.normal(0, 0.5)
        # noise = 0
        act = np.clip(act + noise, self.action_low, self.action_high)
        return act

    def train(self, ep):
        batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, 2))
        actions = np.zeros((self.batch_size, 1))
        rewards = np.zeros((self.batch_size, 1))
        
        for i, sample in enumerate(batch):

            states[i] = sample[0]   #shape = (1, 2)
            actions[i] = sample[1]  #shape = (1, 1)
            rewards[i] = sample[2]  #shape = (,)


        pred_actions = self.target_actor.predict(states)
        self.actor_update([states, pred_actions])
        self.critic_update([states, actions, rewards])
        # action_grad = self.get_action_gradients(states, pred_action)

    def append_sample(self, states, actions, rewards):
        for i in range(len(states)):
            self.memory.append((states[i], actions[i], rewards[i]))

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
    agent.load_model('ddpg')
    # EPISODES = 500000
    if not os.path.exists('save_model'):
        os.makedirs('save_model')
    e = 0
    if os.path.exists('ddpg_output.csv'):
        with open('ddpg_output.csv', 'r') as f:
            read = csv.reader(f)
            e = int(next(reversed(list(read)))[0])
        print(e)
    stats = []

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmax', type=int, default=1000)
    parser.add_argument('--lambd', type=float, default=1.0)
    args = parser.parse_args()
    TMAX = args.tmax
    LAMBDA = args.lambd

    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    action_examples = np.array([env.action_space.sample() for x in range(10000)])
    scaler_s = StandardScaler()
    scaler_a = StandardScaler()
    scaler_s.fit(observation_examples)
    scaler_a.fit(action_examples)

    np.set_printoptions(precision=4, suppress=True)

    while True:
        done = False
        step = 0
        T = 0
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        state = list(scaler_s.transform(state)[0])
        state = np.reshape(state, [1, agent.state_size])
        state_list, action_list, reward_list = [], [], []

        while not done:
            if RENDER and e > 20:
                env.render()
            if TRAIN:
                if len(agent.memory) >= 2000:
                    agent.train(e)

            action = agent.get_action(state, e)
            next_state, reward, done, _ = env.step(action)
            T += 1
            step += 1
            score += reward
            if step % 10 == 0 or done:
                print(np.float32(action), np.float32(reward), np.float32(score), end='\r')
            
            next_state = np.reshape(next_state, [1, agent.state_size])
            next_state = list(scaler_s.transform(next_state)[0])
            next_state = np.reshape(next_state, [1, agent.state_size])
            action = scaler_a.transform(np.reshape(action, (-1, 1)))[0]

            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)

            state = next_state


            if T >= TMAX or done:
                T = 0
                states = state_list + [next_state]
                states = np.stack(states, axis=1)
                states = states.reshape(states.shape[1:])
                actions = agent.target_actor.predict(states)
                Qs = agent.target_critic.predict([states, actions])
                discounted = np.zeros_like(reward_list)
                if done:
                    G_LAMBD = 0
                else:
                    G_LAMBD = Qs[-1]

                for t in reversed(range(len(reward_list))):
                    # G = reward_list[t] + agent.gamma * G
                    G_LAMBD = reward_list[t] + ( agent.gamma *
                                (LAMBDA * G_LAMBD + (1 - LAMBDA) * Qs[t+1])
                    )
                    discounted[t] = G_LAMBD

                agent.append_sample(state_list, action_list, discounted)
                state_list, action_list, reward_list = [], [], []


            if done:
                # log
                e += 1
                if TRAIN:
                    agent.update_target_model()
                    print(e, 'episode trained')

                    stat = [e, step, score]
                    stats.append(stat)

                    with open('ddpg_output.csv', 'a', encoding='utf-8', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow(stat)
                    if e % 20 == 0:
                        agent.save_model('ddpg')
                        stats.clear()
                else:                    
                    print('step: %s, \t\t\tscore: %s' % (step, score))
