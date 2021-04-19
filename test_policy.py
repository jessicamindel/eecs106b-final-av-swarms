from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
from deep_rl_for_swarms.common.act_wrapper import ActWrapper
from deep_rl_for_swarms.ma_envs.envs.point_envs import rendezvous

import pdb
import time
import numpy as np

POLICY_FILENAME = '/tmp/baselines/trpo_test/rendezvous/20210418_2004_40/model.pkl'

def main():
    # WARNING: This must match the class of the saved policy. See the main() method in train.py
    def policy_fn(name, ob_space, ac_space):
        return mlp_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                    hid_size=[64], feat_size=[64])

    # Load the policy
    po = ActWrapper.load(POLICY_FILENAME, policy_fn)

    # WARNING: This must match the environment for the saved policy. See the main() method of train.py
    env = rendezvous.RendezvousEnv(nr_agents=20,
                                   obs_mode='sum_obs_acc',
                                   comm_radius=100 * np.sqrt(2),
                                   world_size=100,
                                   distance_bins=8,
                                   bearing_bins=8,
                                   torus=False,
                                   dynamics='unicycle_acc')

    obs = env.reset()

    stochastic = True # idk why not
    while True:
        ac, vpred = po._act.act(stochastic, obs)
        obs, reward, done, info = env.step(ac)
        env.render()

if __name__ == "__main__":
    main()