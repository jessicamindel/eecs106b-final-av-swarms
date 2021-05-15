from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from deep_rl_for_swarms.common.act_wrapper import ActWrapper
from deep_rl_for_swarms.ma_envs.envs.point_envs import rendezvous

import pdb
import time
import numpy as np
import argparse
from sim import Sim
import matplotlib.pyplot as plt
import scipy

POLICY_FILES = {
    'maps/task1a_moreborders.png': "policies/task1a_moreborders.png20210510_2138_44.pkl",
    'maps/task2a_moreborders.png': "policies/task2a_moreborders.png20210510_2139_17.pkl",
    'maps/task4_moreborders.png': "policies/task4_moreborders.png20210510_2139_28.pkl",
    'maps/urbanish.png': "policies/task2a_moreborders.png20210510_2139_17.pkl"
}

def main():
    parser = argparse.ArgumentParser(description='Run autonomous vehicle swarm simulation.')
    parser.add_argument('num_cars', type=int)
    parser.add_argument('map_path')
    parser.add_argument('--path-reversal-prob', type=float, required=False)
    parser.add_argument('--angle-min', type=float, required=False)
    parser.add_argument('--angle-max', type=float, required=False)
    parser.add_argument('--timestep', type=float, required=False, default=0.1)
    parser.add_argument('--angle-mode', choices=['auto', 'auto_noise', 'random'], default='auto', required=False)
    parser.add_argument('--angle-noise', type=float, default=0.0, required=False)
    parser.add_argument('--save-video', action='store_true', default=False, required=False)
    parser.add_argument('--nogui', action='store_true', default=False, required=False)
    parser.add_argument('--collision-penalty', choices=['none', 'low'], default='none', required=False)
    parser.add_argument('--num_episodes', type=int, default=1, required=False)

    args = parser.parse_args()

    POLICY_FILENAME = POLICY_FILES[args.map_path]

    if args.save_video:
        assert not args.nogui

    # WARNING: This must match the class of the saved policy. See the main() method in train.py
    def policy_fn(name, ob_space, ac_space):
        return mlp_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                    hid_size=[64], feat_size=[64])

    # Load the policy
    po = ActWrapper.load(POLICY_FILENAME, policy_fn)

    # WARNING: This must match the environment for the saved policy. See the main() method of train.py
    env = Sim(
        args.num_cars, args.map_path, args.path_reversal_prob or 0,
        args.angle_min or 0, args.angle_max or 2*np.pi,
        angle_mode=args.angle_mode, angle_noise=args.angle_noise,
        timestep=args.timestep, save_video=args.save_video, 
        collision_penalty=args.collision_penalty
    )

    if not args.nogui:
        # Create window for rendering
        plt.ion()
        fig, ax = plt.subplots(figsize=(8,8))
        fig.canvas.set_window_title('AV Swarm Simulator')
        plt.show()
        env.render(ax=ax)
        plt.pause(0.01)

    stochastic = True # idk why not
    returns = []
    collisions = []
    goals = []
    for i in range(args.num_episodes):
        obs = env.reset()
        done = False
        ep_return = 0
        while not done:
            ac, vpred = po._act.act(stochastic, obs)
            obs, reward, done, info = env.step(ac)
            ep_return += reward
            if not args.nogui:
                env.render(ax=ax)
                plt.pause(0.01)

        collisions.append(env.check_collisions())
        goals.append(env.goals_reached)
        returns.append(ep_return)
        if i % 10 == 0:
            print(i)

    returns = np.array(returns).sum(axis=1) / args.num_cars
    print(f"avg returns ${np.sum(returns)/len(returns):.2f}\\pm{scipy.stats.sem(returns):.2f}$")
    print(f"avg goals ${np.sum(goals)/len(goals):.2f}\\pm{scipy.stats.sem(goals):.2f}$")
    print(f"avg collisions ${np.sum(collisions)/len(collisions):.2f}\\pm{scipy.stats.sem(collisions):.2f}$")

    env.close()

if __name__ == "__main__":
    main()