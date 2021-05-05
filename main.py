import numpy as np
import argparse
from sim import Sim
import matplotlib.pyplot as plt

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run autonomous vehicle swarm simulation.')
	parser.add_argument('num_cars', type=int)
	parser.add_argument('map_path')
	parser.add_argument('--path-reversal-prob', type=float, required=False)
	parser.add_argument('--angle-min', type=float, required=False)
	parser.add_argument('--angle-max', type=float, required=False)
	parser.add_argument('--save-video', action='store_true', default=False, required=False)
	parser.add_argument('--timestep', type=float, required=False, default=0.1)

	args = parser.parse_args()
	s = Sim(args.num_cars, args.map_path, args.path_reversal_prob or 0, args.angle_min or 0, args.angle_max or 2*np.pi, save_video=args.save_video, timestep=args.timestep)
	s.render()
	plt.pause(0.01)

	# ACTIONS = [(200, 0)] #, (0, np.pi/6), (50, np.pi/12)]
	ACTIONS = [(10, np.pi/6)]
	rng = np.random.default_rng(42)
	car_actions = [rng.choice(ACTIONS) for _ in range(args.num_cars)]

	for i in range(80):
		s.step(car_actions)
		s.render()
		plt.pause(0.01)

	plt.pause(15)
	s.close()

	# FIXME: Some problems right off the bat:
	# - I can't keep the plt window open without pausing it for a long period of time.
