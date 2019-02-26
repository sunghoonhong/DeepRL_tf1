'''
Author: Sunghoon Hong
Title: play_ra3c.py
Version: 0.0.1
Description: Play RA3C Agent
Detail:
    Sequence size = 2
    Action size = 4
    Weight of Entropy of actor loss = 0.1
    Loss function of Critic = Huber loss
    Change lr to 5e-4
    4-step TD
    2-layer CNN
    LSTM output = 512
    Modify discounted_prediction()

'''

import os   
import time
import random
import numpy as np
from PIL import ImageOps, Image
from matplotlib import pyplot as plt
from snake_env import Env, ACTION
from ra3c_agent import A3CAgent

global episode
episode = 0
if not os.path.exists('save_graph/ra3c_agent'):
    os.makedirs('save_graph/ra3c_agent')
if not os.path.exists('save_model'):
    os.makedirs('save_model')

EPISODES = 3
DELAY = 0.1
IMPROVE = 'policy'

if __name__ == "__main__":
    global_agent = A3CAgent(
        load_model=True,
        verbose=False,
        render=True
    )
    global_agent.play(EPISODES, DELAY, IMPROVE)
