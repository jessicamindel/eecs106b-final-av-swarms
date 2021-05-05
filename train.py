"""
Requires you to clone https://github.com/ALRhub/deep_rl_for_swarms
and install it as a pip library

See the README for installation
"""

# Based on https://github.com/ALRhub/deep_rl_for_swarms/blob/master/deep_rl_for_swarms/run_multiagent_trpo.py

#!/usr/bin/env python3
import datetime
import numpy as np
from mpi4py import MPI
from deep_rl_for_swarms.common import logger
from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from deep_rl_for_swarms.rl_algo.trpo_mpi import trpo_mpi

import argparse
from sim import Sim
# from deep_rl_for_swarms.ma_envs.envs.point_envs import rendezvous
import pdb


def train(num_timesteps, log_dir):
    parser = argparse.ArgumentParser(description='Run autonomous vehicle swarm simulation.')
    parser.add_argument('num_cars', type=int)
    parser.add_argument('map_path')
    parser.add_argument('--path-reversal-prob', type=float, required=False)
    parser.add_argument('--angle-min', type=float, required=False)
    parser.add_argument('--angle-max', type=float, required=False)
    parser.add_argument('--timestep', type=float, required=False, default=0.1)
    parser.add_argument('--angle-mode', choices=['auto', 'auto_noise', 'random'], default='auto', required=False)
    parser.add_argument('--angle-noise', type=float, default=0.0, required=False)

    args = parser.parse_args()

    import deep_rl_for_swarms.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure(format_strs=['csv'], dir=log_dir)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    def policy_fn(name, ob_space, ac_space):
        return mlp_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                   hid_size=[64], feat_size=[64])

    env = Sim(
        args.num_cars, args.map_path, args.path_reversal_prob or 0,
        args.angle_min or 0, args.angle_max or 2*np.pi,
        angle_mode=args.angle_mode, angle_noise=args.angle_noise,
        timestep=args.timestep
    )
    # env = rendezvous.RendezvousEnv(nr_agents=20,
    #                                obs_mode='sum_obs_acc',
    #                                comm_radius=100 * np.sqrt(2),
    #                                world_size=100,
    #                                distance_bins=8,
    #                                bearing_bins=8,
    #                                torus=False,
    #                                dynamics='unicycle_acc')

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=2048, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()


def main():
    env_id = "Rendezvous-v0"
    dstr = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')
    log_dir = '/tmp/baselines/trpo_test/rendezvous/' + dstr
    train(num_timesteps=1e7, log_dir=log_dir)


if __name__ == '__main__':
    main()