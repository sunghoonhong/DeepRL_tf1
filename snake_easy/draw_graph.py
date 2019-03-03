import os
import csv
import numpy as np
from matplotlib import pyplot as plt

agent = 'ra3c'
name = agent+'_agent'
filename = agent+'_output.csv'

if not os.path.exists('save_graph/'+name):
    os.makedirs('save_graph/'+name)

def smooth(arr, n):
    end = -(len(arr)%n)
    if end == 0:
      end = None
    arr = np.reshape(arr[:end], (-1, n))
    arr = np.mean(arr, axis=1)
    return arr

def draw(x, y, ylabel, dir='save_graph/'+name+'/'):
    plt.figure(figsize=(15, 5))
    plt.plot(x, y)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.savefig(dir+ylabel+'.png')
    # plt.show()
    plt.clf()

def drawall(n=100, begin=0):
    es = smooth(episodes[-begin:], n)
    step = smooth(steps[-begin:], n)
    reward = smooth(rewards[-begin:], n)
    score = smooth(scores[-begin:], n)
    over4 = smooth([int(i>4) for i in scores[-begin:]], n)
    ps = smooth(pmaxs[-begin:], n)
    al = smooth(aloss[-begin:], n)
    cl = smooth(closs[-begin:], n)
    info  = smooth(infos[-begin:], n)
    draw(es, reward, 'reward')
    draw(es, score, 'score')
    draw(es, step, 'step')
    draw(es, over4, 'over 4 score')
    draw(es, ps, 'p max')
    draw(es, al, 'actor loss')
    draw(es, cl, 'critic loss')
    draw(es, info, 'self collision')

if __name__=='__main__':
    episodes = []
    scores = []
    rewards = []
    steps = []
    pmaxs = []
    aloss = []
    closs = []
    infos = []
    with open(filename, 'r') as f:
        read = csv.reader(f)
        for i, row in enumerate(read):
            episodes.append(i)
            steps.append(int(float(row[1])))
            rewards.append(float(row[2]))
            scores.append(int(float(row[3])))
            pmaxs.append(float(row[4]))
            aloss.append(float(row[5]))
            closs.append(float(row[6]))
            infos.append(1 if row[7]=='body' else 0)
    drawall(n=100)