import matplotlib.pyplot as plt
import numpy as np

from utils import *

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

    TASK1_FILE = '/Users/himty/Downloads/maps/task1a_moreborders.png20210510_2138_44/progress.csv'

    import csv
    with open(TASK1_FILE, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            print(', '.join(row))

    # data = data4_forward
    # plt.plot(list(range(len(data))), data)
    # plt.title("Episode Return Means Over Rounds of TRPO")
    # plt.xlabel("Rounds")
    # plt.ylabel("Episode Reward Mean")
    # plt.show()

if __name__ == "__main__": main(),
