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

def drawall(name, x, metrics, labels, n=100, begin=0):
  dir ='save_graph/%s'% name
  if not os.path.exists(dir):
    os.makedirs(dir)
  
  x = smooth(x[-begin:], n)
  for i, metric in enumerate(metrics):
    metrics[i] = smooth(metric[-begin:], n)

  def draw(x, y, ylabel):
    plt.figure(figsize=(15,5))
    plt.plot(x, y)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.savefig(dir+'/'+ylabel)
    plt.clf()

  for i, metric in enumerate(metrics):
    draw(x, metric, labels[i])


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
    boundarys = []
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
            closs.append(float(row[6]))
            self_colls.append(1 if row[7]=='body' else 0)
            boundarys.append(1 if row[7]=='boundary' else 0)
            timeouts.append(1 if row[7]=='timeout' else 0)
            step_per_score.append(steps[i] / scores[i] if scores[i] > 0 else steps[i])

    metrics = [
      rewards,
      scores,
      steps,
      pmaxs,
      over4,
      self_colls,
      boundarys,
      step_per_score,
      timeouts,
      aloss,
      closs
    ]
    labels = [
      'reward',
      'score',
      'step',
      'pmax',
      'over 4 score',
      'self collision',
      'boundary',
      'step per score',
      'timeout',
      'actor loss',
      'critic loss'
    ]
    drawall(name, episodes, metrics, labels, args.n, args.begin)
