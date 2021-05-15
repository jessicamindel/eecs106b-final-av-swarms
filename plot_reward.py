import matplotlib.pyplot as plt
import numpy as np

from utils import *
import csv

def main():
    """
    PHI_MIN = -np.pi/2 * 0.6,
    PHI_MAX = np.pi/2 * 0.6,
,
    V_MIN = -10,
    V_MAX = 100,
,
    DPHI_MIN = -np.pi/30,
    DPHI_MAX = np.pi/30,
,
    V_MANUAL_INCREMENT = 5.0,
    DPHI_MANUAL_INCREMENT = 0.001,
,
    TIMESTEP = 0.05,
,
    CAR_L = 5,
    CAR_LEN = 6,
    CAR_W = 3,
    CAR_COLLIDER_BOUND2 = CAR_LEN * CAR_LEN + CAR_W * CAR_W,
,
    LIDAR_MIN = -np.pi/2,
    LIDAR_MAX = np.pi/2,
    LIDAR_N = 10,
,
    DPHI_PENALTY_THRESHOLD = np.pi/200 # FIXME: May be too small or large?,
    DPHI_PENALTY_MAX = np.pi/40 # FIXME: May be too small or large?,
,
    def func(dphi):,
      return 20*lerp(normalize_between(np.abs(dphi), DPHI_PENALTY_THRESHOLD, DPHI_PENALTY_MAX), 0, 1/200),

    dphis = np.linspace(PHI_MIN, PHI_MAX, 100),

    plt.plot(dphis, [func(dphi) for dphi in dphis]),
    plt.xlabel('dphi'),
    plt.ylabel('Reward'),
    plt.show(),
    """

    conf = {
        'task1': {
            'file': '/Users/himty/Downloads/maps/task1a_moreborders.png20210510_2138_44/progress.csv',
        },
        'task2': {
            'file': '/Users/himty/Downloads/maps/task2a_moreborders.png20210510_2139_17/progress.csv',
        },
        'task3': {
            'file': '/Users/himty/Downloads/maps/task4_moreborders.png20210510_2139_28/progress.csv',
        },

        'tasks': ['task1', 'task2', 'task3'],
        'columns': ['EpRewMean', "EpisodesSoFar"]
    }

    data = {}
    for task in conf['tasks']:
        data[task] = {col: [] for col in conf['columns']}

        with open(conf[task]['file'], newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

            headers = next(spamreader)[0].split(',')
            col_indices = [headers.index(col) for col in conf['columns']]
            for row in spamreader:
                for col, col_idx in zip(conf['columns'], col_indices):
                    data[task][col].append(float(row[0].split(',')[col_idx]))

    for task, task_data in data.items():
        plt.plot(task_data['EpisodesSoFar'], task_data['EpRewMean'], label=task)
    plt.title("Episode Return Means Episodes Used in TRPO")
    plt.xlabel("Episodes")
    plt.ylabel("Episode Reward Mean")
    plt.legend()
    plt.show()

if __name__ == "__main__": main(),
