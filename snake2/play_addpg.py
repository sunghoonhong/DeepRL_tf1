'''
Author: Sunghoon Hong
Title: play_addpg.py
Version: 0.0.1
Description: Play ADDPG Agent
Detail:

'''

import os   
import time
import random
import numpy as np
from PIL import ImageOps, Image
from snake2_env import Env
from addpg_agent import ADDPGAgent

global episode
episode = 0
if not os.path.exists('save_graph/ra3c_agent'):
    os.makedirs('save_graph/ra3c_agent')
if not os.path.exists('save_model'):
    os.makedirs('save_model')

DEBUG = False
EPISODES = 3
DELAY = 0.1
IMPROVE = 'policy'
# IMPROVE = 'exploration'

if __name__ == "__main__":
    env = Env()
    global_agent = ADDPGAgent(
        action_space=env.action_space,
        load_model=True,
        render=True
    )
    global_agent.play(EPISODES, DELAY, IMPROVE, DEBUG)
