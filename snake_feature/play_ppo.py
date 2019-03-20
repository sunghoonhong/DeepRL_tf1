'''
Author: Sunghoon Hong
Title: play_ra3c.py
Version: 0.0.1
Description: Play A3C Agent
Detail:
'''

import os   
import time
import random
import argparse
import numpy as np
from keras.models import Sequential, Model
from keras.layers import TimeDistributed, Reshape, Input
from keras.layers import Conv2D, Flatten, Dense, GRU, BatchNormalization
import tensorflow as tf
from PIL import ImageOps, Image
from snake_env import Env, ACTION

global episode
episode = 0
if not os.path.exists('save_graph/ppo'):
    os.makedirs('save_graph/ppo')
if not os.path.exists('save_model'):
    os.makedirs('save_model')


class Agent:
    def __init__(self, state_size, action_size, seq_size, 
                verbose, load_model, render):
        self.render = render
        self.state_size = state_size
        self.state_shape = [1] + self.state_size
        self.action_size = action_size

        self.actor, self.critic = self.build_model()
        if load_model:
            self.load_model('./save_model/ppo')

    def build_model(self, actor_only=False):
        state = Input(shape=self.state_size)
        state_process = Dense(100, activation='elu')(state)
        state_process = BatchNormalization()(state_process)
        state_process = Dense(100, activation='elu')(state_process)
        state_process = BatchNormalization()(state_process)
        policy = Dense(self.action_size, activation='softmax')(state_process)
        value = Dense(1, activation='linear', kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(state_process)

        actor = Model(inputs=state, outputs=policy, name='Actor')
        critic = Model(inputs=state, outputs=value, name='Critic')
        actor._make_predict_function()
        critic._make_predict_function()
        
        return actor, critic

    def load_model(self, name):
        if os.path.exists(name + '_actor.h5'):
            self.actor.load_weights(name + '_actor.h5')
            print('Actor loaded')
        else:
            print('No actor')
        if os.path.exists(name + '_critic.h5'):
            self.critic.load_weights(name + '_critic.h5')
            print('Critic loaded')
        else:
            print('No critic')
        

    def get_action(self, state):
        policy = self.actor.predict(state)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--delay', type=float, default=0.)
    parser.add_argument('--episode', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--seqsize', type=int, default=2)
    args = parser.parse_args()
    keymap = {'w': 0 , 'a': 1, 'd': 2}
    
    env = Env()
    agent = Agent(
        state_size=env.state_size,
        action_size=env.action_size,
        seq_size=args.seqsize,
        load_model=args.load_model,
        verbose=args.verbose,
        render=args.render
    )
    np.set_printoptions(precision=4, suppress=True)
    
    for e in range(args.episode):
        step = 0
        reward_sum = 0
        done = False
        observe, _, _, _ = env.reset()

        state = observe.reshape(agent.state_shape) / 20.

        while not done:
            step += 1
            if agent.render:
                env.render()
            time.sleep(args.delay)
            action, policy = agent.get_action(state)
            if args.debug:
                print('St:', state, 'V(St):', agent.critic.predict(state))
                print('At:', ACTION[action], policy)
                while True:
                    key = input('Press y or action(w, a, d): ')
                    if key == 'y':
                        break
                    elif key in keymap:
                        action = keymap[key]
                        break
            real_action = action
            next_observe, reward, done, info = env.step(real_action)
            next_state = next_observe.reshape(agent.state_shape) / 20.
            if args.debug:
                print('St+1:', next_state, 'Rt+1:', reward, info)

            reward_sum += reward
            state = next_state

        episode += 1
        print('Ep', e, 'Score', env.game.score, 'Step', step)