"""
Requires you to clone https://github.com/ALRhub/deep_rl_for_swarms
and install it as a pip library

cd deep_rl_for_swarms
brew install mpich # for mpi4py. I'm on a Mac
# Open setup.py
# - Change "tensorboard == 1.5.0" to "tensorboard"
# - Change "tensorflow == 1.5.2" to "tensorflow"
# - Change "numpy == 1.14.5" to "numpy" # Too low for the fast_histogram package
pip install -e .

MacOS fix for Matplotlib's "RuntimeError: Python is not installed as a framework"
https://stackoverflow.com/questions/34977388/matplotlib-runtimeerror-python-is-not-installed-as-a-framework
In the file ~/.matplotlib/matplotlibrc, add the line "backend: TkAgg"

In deep_rl_for_swarms/deep_rl_for_swarms/common/act_wrapper.py change
"from deep_rl_for_swarms.common import logger" to "from deep_rl_for_swarms.common import logger"

To save the trained policy approximately every hour:
On deep_rl_for_swarms/rl_algo/trpo_mpi/trpo_mpi.py line 325, right above iters_so_far += 1, add
if iters_so_far % 1000 == 0 and iters_so_far > 0:
    pi.save()
"""

# Based on https://github.com/ALRhub/deep_rl_for_swarms/blob/master/deep_rl_for_swarms/run_multiagent_trpo.py

#!/usr/bin/env python3
import datetime
import numpy as np
from mpi4py import MPI
from deep_rl_for_swarms.common import logger
from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from deep_rl_for_swarms.rl_algo.trpo_mpi import trpo_mpi
from deep_rl_for_swarms.ma_envs.envs.point_envs import rendezvous
import pdb


def train(num_timesteps, log_dir):
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

    env = rendezvous.RendezvousEnv(nr_agents=20,
                                   obs_mode='sum_obs_acc',
                                   comm_radius=100 * np.sqrt(2),
                                   world_size=100,
                                   distance_bins=8,
                                   bearing_bins=8,
                                   torus=False,
                                   dynamics='unicycle_acc')

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