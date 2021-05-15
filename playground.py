import numpy as np
from sim import Sim, ManualCar
import matplotlib.pyplot as plt
import argparse
import scipy

POLICY_FILES = {
	'task1': "policies/task1a_moreborders.png20210510_2138_44.pkl",
	'task2': "policies/task2a_moreborders.png20210510_2139_17.pkl",
	'task3': "policies/task4_moreborders.png20210510_2139_28.pkl",
	'task4': "policies/task4_moreborders.png20210510_2139_28.pkl"
}

MAP_FILES = {
	'task1': "maps/task1a_moreborders.png",
	'task2': "maps/task2a_moreborders.png",
	'task3': "maps/task4_moreborders.png",
	'task4': "maps/urbanish.png"
}

NUM_RL_CARS = 3

def get_policy_action_func(policy_filename):
	from deep_rl_for_swarms.policies import mlp_mean_embedding_policy
	from deep_rl_for_swarms.common.act_wrapper import ActWrapper
	# WARNING: This must match the class of the saved policy. See the main() method in train.py
	def policy_fn(name, ob_space, ac_space):
		return mlp_mean_embedding_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
													hid_size=[64], feat_size=[64])
	# Load the policy
	po = ActWrapper.load(policy_filename, policy_fn)

	def get_action(obs):
		ac, vpred = po._act.act(True, obs)
		return ac
	return get_action

def get_random_action_func():
	ACTIONS = [(100, 0), (50, 0.02), (50, -0.02)]
	car_actions = [rng.choice(ACTIONS) for _ in range(NUM_RL_CARS)]
	def get_action(obs):
		return car_actions
	return get_action

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run autonomous vehicle swarm simulation.')
	parser.add_argument('task', type=str, choices=['task1', 'task2', 'task3', 'task4'])
	parser.add_argument('other_cars_type', type=str, choices=['random', 'policy'])
	parser.add_argument("control_car_type", type=str, choices=['random', 'human'])
	parser.add_argument('--num_episodes', type=int, default=1, required=False)
	parser.add_argument('--save-video', action='store_true', default=False, required=False)
	parser.add_argument('--nogui', action='store_true', default=False, required=False)

	args = parser.parse_args()

	policy_filename = POLICY_FILES[args.task]

	rng = np.random.default_rng(42)
	if args.other_cars_type == "random":
		get_car_actions = get_random_action_func()
	elif args.other_cars_type == "policy":
		get_car_actions = get_policy_action_func(policy_filename)
	else:
		raise ValueError()

	if not args.nogui:
		plt.ion()
		fig, ax = plt.subplots(figsize=(8,8))
		fig.canvas.set_window_title('AV Swarm Simulator')
		plt.show()

	s = Sim(NUM_RL_CARS, MAP_FILES[args.task], save_video=args.save_video)

	returns = []
	collisions = []
	goals = []
	for i in range(args.num_episodes):
		done = False
		ep_return = 0
		obs = s.reset() # Do this BEFORE adding a manual or random car

		if args.control_car_type == "human":
			s.add_manual_car(fig)
		elif args.control_car_type == "random":
			s.add_random_car()

		obs = s.get_obs()

		if not args.nogui:
			s.render(ax=ax)
			plt.pause(0.01)

		while not done:
			obs, reward, done, info = s.step(get_car_actions(obs))
			ep_return += reward
			if not args.nogui:
				s.render(ax=ax)
				plt.pause(0.01)
		if i % 10 == 0:
			print(i)

		collisions.append(s.check_collisions())
		goals.append(s.goals_reached)
		returns.append(ep_return)

	returns = np.array(returns).sum(axis=1) / NUM_RL_CARS
	print(f"avg returns ${np.sum(returns)/len(returns):.2f}\\pm{scipy.stats.sem(returns):.2f}$")
	print(f"avg goals ${np.sum(goals)/len(goals):.2f}\\pm{scipy.stats.sem(goals):.2f}$")
	print(f"avg collisions ${np.sum(collisions)/len(collisions):.2f}\\pm{scipy.stats.sem(collisions):.2f}$")

	s.close()
