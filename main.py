import numpy as np
import argparse
from sim import Sim

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run autonomous vehicle swarm simulation.')
	parser.add_argument('num_cars', type=int)
	parser.add_argument('map_path')
	parser.add_argument('--path_reversal_prob', type=float, required=False)
	parser.add_argument('--angle_min', type=float, required=False)
	parser.add_argument('--angle_max', type=float, required=False)

	args = parser.parse_args()
	s = Sim(args.num_cars, args.map_path, args.path_reversal_prob or 0, args.angle_min or -np.pi, args.angle_max or np.pi)
