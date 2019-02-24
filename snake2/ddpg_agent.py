
import os
import csv
import random
from collections import deque
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

RESIZE = 84
SEQ_SIZE = 4

VERBOSE = False
# RENDER = True
RENDER = False
# TRAIN = False
TRAIN = True


def preprocess(observe):
    ret = Image.fromarray(observe)
    ret = ImageOps.mirror(ret.rotate(270)).convert('L').resize((RESIZE, RESIZE))
    return np.asarray(ret)


class DDPGAgent:
    def __init__(self, action_space, render=False):
        self.render = render

        self.seq_size = SEQ_SIZE
        self.state_size = (self.seq_size, RESIZE, RESIZE)
        self.action_size = action_space.shape[0]
        self.action_low = action_space.low
        self.action_high = action_space.high
        
        # Hyper-Parameter
        self.actor_lr = 1e-6
        self.critic_lr = 2.5e-5
        self.gamma = 0.99
        self.memory_size = 400000
        self.train_start = 50000
        self.update_target_rate = 20000
        self.batch_size = 32
        # self.good_batch_size = 4
        # self.normal_size = self.batch_size - self.good_batch_size

        # TF Session
        self.sess = tf.Session()
        K.set_session(self.sess)

        # Model
        self.actor, self.critic = self.build_model()
        self.target_actor, self.target_critic = self.build_model()
        self.update_target_model()
        self.action_grad = tf.gradients(self.critic.output, self.critic.input[1])
        self.actor_optimize = self.actor_optimizer()
        self.critic_update = self.critic_optimizer()

        self.sess.run(tf.global_variables_initializer())

        # Replay Memory
        self.memory = deque(maxlen=self.memory_size)
        # self.good_memory = deque(maxlen=self.memory_size)

    def build_model(self):
        state_size = list(self.state_size)
        state_size.append(1)
        state = Input(shape=self.state_size)
        reshape = Reshape(state_size)(state)

        conv = TimeDistributed(Conv2D(16, (8, 8), strides=(4, 4), activation='relu'))(reshape)
        conv = TimeDistributed(Conv2D(32, (4, 4), strides=(2, 2), activation='relu'))(conv)
        conv = TimeDistributed(Flatten())(conv)
        
        lstm_state = LSTM(512, activation='relu')(conv)
        action_output = Dense(self.action_size, activation='tanh')(lstm_state)
        actor_output = Lambda(lambda x: x * np.pi)(action_output)

        actor = Model(inputs=state, outputs=action_output)

        action = Input([self.action_size])
        state_action = Concatenate()([lstm_state, action])
        fc = Dense(512, activation='relu')(state_action)
        Q_output = Dense(1)(fc)
    
        critic = Model(inputs=[state, action], outputs=Q_output)

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
        optimizer = tf.train.AdamOptimizer(self.actor_lr)
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
        optimizer = Adam(lr=self.critic_lr)
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
        noise = np.random.normal()
        act = np.clip(act + noise, self.action_low, self.action_high)
        return act

    def train(self):
        batch = random.sample(self.memory, self.batch_size)
        # good_batch = random.sample(self.good_memory, self.good_batch_size)

        states = np.zeros([self.batch_size] + list(self.state_size))
        actions = np.zeros((self.batch_size, 1))
        rewards = np.zeros((self.batch_size, 1))
        next_states = np.zeros([self.batch_size] + list(self.state_size))
        dones = np.zeros((self.batch_size, 1))
        
        for i, sample in enumerate(batch):

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
    env = Env()
    agent = DDPGAgent(env.action_space, RENDER)
    agent.load_model('ddpg')
    # EPISODES = 500000
    if not os.path.exists('save_model'):
        os.makedirs('save_model')
    episode = 0
    if os.path.exists('output.csv'):
        with open('output.csv', 'r') as f:
            read = csv.reader(f)
            episode = int(next(reversed(list(read)))[0])
        print(episode)
    stats = []

    while True:
        step = 0
        global_step = 0
        reward_sum = 0
        done = False
        observe, _, _, _ = env.reset()

        state = preprocess(observe).reshape((1, RESIZE, RESIZE))
        state = np.float32(state / 255.)
        history = np.copy(state)
        for _ in range(agent.seq_size - 1):
            history = np.append(history, state, axis=0)
            state = np.copy(state)
        history = np.reshape([history], (1, agent.seq_size, RESIZE, RESIZE))

        while not done:
            if agent.render:
                env.render()
            step += 1
            global_step += 1

            action = agent.get_action(history)[0][0]
            print(action)   #
            next_observe, reward, done, info = env.step(action)
            next_state = preprocess(next_observe)
            next_state = np.reshape([next_state], (1, 1, RESIZE, RESIZE))
            next_state = np.float32(next_state / 255.)
            next_history = np.append(next_state, history[:, :(agent.seq_size-1), :, :], axis=1)

            agent.append_sample(history, action, reward, next_history, done)

            if len(agent.memory) >= agent.train_start:
                agent.train()

            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            history = next_history

            # statistics
            reward_sum += reward

        # stats.append((e, env.game.score, step, info))
        episode += 1
        stats.append([episode, step, reward_sum, env.game.score, info])

        if episode % SAVE_EPISODE_RATE == 0:
            with open('ddpg_output.csv', 'a', encoding='utf-8', newline='') as f:
                wr = csv.writer(f)
                for row in stats:
                    wr.writerow(row)
            mean = np.mean(np.float32(np.transpose(stats)[:4]), axis=1)
            print('Ep %s Step: %s Reward: %s Score: %s' % (episode, mean[1], mean[2], mean[3]), end='\r')
            stats.clear()
            agent.save_model('ddpg')
            