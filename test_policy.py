from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from deep_rl_for_swarms.common.act_wrapper import ActWrapper
from deep_rl_for_swarms.ma_envs.envs.point_envs import rendezvous

import pdb
import time
import numpy as np
import argparse
from sim import Sim
import matplotlib.pyplot as plt

# POLICY_FILENAME = '/tmp/baselines/trpo_test/rendezvous/20210418_2004_40/model.pkl'
# POLICY_FILENAME = '/tmp/baselines/trpo_test/rendezvous/20210505_1219_46/model.pkl'
POLICY_FILENAME = '/Users/himty/Downloads/maps/task4_moreborders.png20210506_1045_33/model.pkl'
# POLICY_FILENAME = '/Users/himty/Downloads/maps/task2a_moreborders.png20210505_2222_11/model.pkl'
# POLICY_FILENAME = '/Users/himty/Downloads/maps/task3_moreborders.png20210505_2315_14/model.pkl'
# POLICY_FILENAME = '/Users/himty/Downloads/maps/task4_moreborders.png20210505_2222_17/model.pkl'

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

    args = parser.parse_args()

    if args.save_video:
        assert not args.nogui

    # WARNING: This must match the class of the saved policy. See the main() method in train.py
    def policy_fn(name, ob_space, ac_space):
        return mlp_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                    hid_size=[64], feat_size=[64])

    # Load the policy
    po = ActWrapper.load(POLICY_FILENAME, policy_fn)

    print('save video?', args.save_video)

    env = Sim(
        args.num_cars, args.map_path, args.path_reversal_prob or 0,
        args.angle_min or 0, args.angle_max or 2*np.pi,
        angle_mode=args.angle_mode, angle_noise=args.angle_noise,
        timestep=args.timestep, save_video=args.save_video, 
        collision_penalty=args.collision_penalty
    )
    # WARNING: This must match the environment for the saved policy. See the main() method of train.py
    # env = rendezvous.RendezvousEnv(nr_agents=20,
    #                                obs_mode='sum_obs_acc',
    #                                comm_radius=100 * np.sqrt(2),
    #                                world_size=100,
    #                                distance_bins=8,
    #                                bearing_bins=8,
    #                                torus=False,
    #                                dynamics='unicycle_acc')

    
    if not args.nogui:
        # Create window for rendering
        plt.ion()
        fig, ax = plt.subplots(figsize=(8,8))
        fig.canvas.set_window_title('AV Swarm Simulator')
        plt.show()
        env.render(ax=ax)
        plt.pause(0.01)

    obs = env.reset()

    stochastic = True # idk why not
    done = False
    while not done:
        ac, vpred = po._act.act(stochastic, obs)
        obs, reward, done, info = env.step(ac)
        print('reward', reward)
        if not args.nogui:
            env.render(ax=ax)
            plt.pause(0.01)
    env.close()

if __name__ == "__main__":
    main()