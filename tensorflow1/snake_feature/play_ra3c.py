'''
Author: Sunghoon Hong
Title: play_ra3c.py
Version: 0.0.1
Description: Play RA3C Agent
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
from PIL import ImageOps, Image
from snake_env import Env, ACTION
from ra3c_agent import preprocess

global episode
episode = 0
if not os.path.exists('save_graph/ra3c_agent'):
    os.makedirs('save_graph/ra3c_agent')
if not os.path.exists('save_model'):
    os.makedirs('save_model')


class Agent:
    def __init__(self, state_size, action_size, seq_size, 
                verbose, load_model, render):
        self.render = render
        self.state_size = state_size
        self.seq_size= seq_size
        self.history_size = [seq_size] + state_size[1:]
        self.action_size = action_size

        self.actor, self.critic = self.build_model()
        if load_model:
            self.load_model('./save_model/ra3c')

    def build_model(self):
        history = Input(shape=self.history_size)
        history_process = TimeDistributed(Dense(256, activation='elu'))(history)
        batch_norm = BatchNormalization()(history_process)
        gru = GRU(256, activation='tanh')(batch_norm)
        policy = Dense(self.action_size, activation='softmax')(gru)
        value = Dense(1, activation='linear')(gru)

        actor = Model(inputs=history, outputs=policy)
        critic = Model(inputs=history, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()

        return actor, critic

    def load_model(self, name):
        if os.path.exists(name + '_actor.h5'):
            self.actor.load_weights(name + '_actor.h5')
            print('Actor loaded')
        if os.path.exists(name + '_critic.h5'):
            self.critic.load_weights(name + '_critic.h5')
            print('Critic loaded')

    def get_action(self, history):
        policy = self.actor.predict(history)[0]
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

        state = preprocess(observe)
        history = np.stack([state] * agent.seq_size, axis=1)

        while not done:
            step += 1
            if agent.render:
                env.render()
            time.sleep(args.delay)
            action, policy = agent.get_action(history)
            if args.debug:
                print('St:', history, 'V(St):', agent.critic.predict(history))
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
            next_state = preprocess(next_observe).reshape([1] + agent.state_size)
            next_history = np.append(history[:, 1:], next_state, axis=1)
            if args.debug:
                print('St+1:', next_history, 'Rt+1:', reward, info)

            reward_sum += reward
            history = next_history

        episode += 1
        print('Ep', e, 'Score', env.game.score, 'Step', step)