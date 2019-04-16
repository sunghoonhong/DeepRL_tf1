import os
import csv
import numpy as np
import argparse
from matplotlib import pyplot as plt


def smooth(arr, n):
    end = -(len(arr)%n)
    if end == 0:
      end = None
    arr = np.reshape(arr[:end], (-1, n))
    arr = np.mean(arr, axis=1)
    return arr

def drawall(name, n=100, begin=0):
  dir ='save_graph/%s'% name
  if not os.path.exists(dir):
    os.makedirs(dir)
  def draw(x, y, ylabel):
    plt.figure(figsize=(15,5))
    plt.plot(x, y)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.savefig(dir+'/'+ylabel)
    plt.clf()
  es = smooth(episodes[-begin:], n)
  score = smooth(scores[-begin:], n)
  reward = smooth(rewards[-begin:], n)
  step = smooth(steps[-begin:], n)
  ps = smooth(pmaxs[-begin:], n)
  # al = smooth(aloss[-begin:], n)
  # cl = smooth(closs[-begin:], n)
  # self_coll = smooth(self_colls[-begin:], n)
  # timeout = smooth(timeouts[-begin:], n)
  # o4 = smooth(over4[-begin:], n)
  # s_per_s = smooth(step_per_score[-begin:], n)

  draw(es, reward, 'reward')
  draw(es, score, 'score')
  draw(es, step, 'step')
  draw(es, ps, 'p max')
  # draw(es, timeout, 'timeout')
  # draw(es, s_per_s, 'step per score')
  # draw(es, o4, 'over4')
  # draw(es, self_coll, 'self collision')
  # draw(es, al, 'actor loss')
  # draw(es, cl, 'critic loss')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--begin', type=int, default=0)
    args = parser.parse_args()

    name = args.agent
    filename = args.agent+'_output.csv'

    episodes = []
    scores = []
    steps = []
    pmaxs = []
    aloss = []
    closs = []
    rewards = []
    self_colls = []
    over4 = []
    step_per_score = []
    timeouts = []

    with open(filename, 'r') as f:
        read = csv.reader(f)
        for i, row in enumerate(read):
            episodes.append(i)
            steps.append(int(float(row[1])))
            rewards.append(float(row[2]))
            scores.append(int(float(row[3])))
            over4.append(1 if scores[i] > 4 else 0)
            pmaxs.append(float(row[4]))
            aloss.append(float(row[5]))
            # closs.append(float(row[6]))
            # self_colls.append(1 if row[7]=='body' else 0)
            # timeouts.append(1 if row[7]=='timeout' else 0)
            # step_per_score.append(steps[i] / scores[i] if scores[i] > 0 else steps[i])
    drawall(name, args.n, args.begin)
