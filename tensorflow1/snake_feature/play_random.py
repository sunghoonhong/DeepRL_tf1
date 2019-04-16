'''
Author: Sunghoon Hong
Title: play_random.py
Version: 0.0.1
Description: Play random Agent
Detail:
'''

import os   
import time
import random
import argparse
import numpy as np
from snake_env import Env, ACTION


def get_action(observe):
    return np.random.choice(range(3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--delay', type=float, default=0.)
    parser.add_argument('--episode', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    keymap = {'w': 0 , 'a': 1, 'd': 2}

    np.set_printoptions(precision=4, suppress=True)
    env = Env()
    for e in range(args.episode):
        done = False
        step = 0
        observe, _, _, _  = env.reset()
        while not done:
            if args.render:
                env.render()
            print('t=', step, observe)
            time.sleep(args.delay)
            step += 1
            action = get_action(observe)
            if args.debug:
                while True:
                    key = input('Press y or action(w, a, d): ')
                    if key == 'y':
                        break
                    elif key in keymap:
                        action = keymap[key]
                        break
            next_observe, reward, done, info = env.step(action)
            if args.debug:
                print(observe, action, next_observe, reward, info)
            
            observe = next_observe

        print('Score:', env.game.score, 'Step:', step, 'Info:', info)
