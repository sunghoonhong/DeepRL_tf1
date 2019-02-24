import os
import csv
import numpy as np
from matplotlib import pyplot as plt

agent = 'ddpg'

def smooth(arr, n):
    end = -(len(arr)%n)
    if end == 0:
      end = None
    arr = np.reshape(arr[:end], (-1, n))
    arr = np.mean(arr, axis=1)
    return arr

def draw(x, y, ylabel):
    plt.figure(figsize=(15, 5))
    plt.plot(x, y)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.savefig(ylabel+'.png')
    plt.show()
    plt.clf()

def drawall(n=100, begin=0):
    es = smooth(episodes[-begin:], n)
    step = smooth(steps[-begin:], n)
    score = smooth(scores[-begin:], n)
    # over4 = smooth([int(i>4) for i in scores[-begin:]], n)
    # ps = smooth(pmaxs[-begin:], n)
    # al = smooth(aloss[-begin:], n)
    # cl = smooth(closs[-begin:], n)
    # info  = smooth(infos[-begin:], n)
    draw(es, score, 'score')
    draw(es, step, 'step')
    # draw(es, over4, 'over 4 score')
    # draw(es, ps, 'p max')
    # draw(es, al, 'actor loss')
    # draw(es, cl, 'critic loss')
    # draw(es, info, 'self collision')

if __name__=='__main__':
    # name = agent+'_agent'
    # filename = agent+'_output.csv'
    filename = 'output.csv'
    episodes = []
    scores = []
    steps = []
    with open(filename, 'r') as f:
        read = csv.reader(f)
        for i, row in enumerate(read):
            episodes.append(i)
            steps.append(int(float(row[1])))
            scores.append(int(float(row[2])))
    drawall(n=20)