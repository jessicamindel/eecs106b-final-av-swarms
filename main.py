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
	parser.add_argument('--nogui', action='store_true', default=False, required=False)
	parser.add_argument('--angle-mode', choices=['auto', 'auto_noise', 'random'], default='auto', required=False)
	parser.add_argument('--angle-noise', type=float, default=0.0, required=False)
	parser.add_argument('--endpoint-mode', choices=['region', 'point'], default='region', required=False)
	parser.add_argument('--collision-penalty', choices=['none', 'low'], default='none', required=False)

	args = parser.parse_args()

	if args.save_video:
		assert not args.nogui

	if not args.nogui:
		# Create window for rendering
		plt.ion()
		fig, ax = plt.subplots(figsize=(8,8))
		fig.canvas.set_window_title('AV Swarm Simulator')
		plt.show()

	s = Sim(
		args.num_cars, args.map_path, args.path_reversal_prob or 0,
		args.angle_min or 0, args.angle_max or 2*np.pi,
		angle_mode=args.angle_mode, angle_noise=args.angle_noise,
		save_video=args.save_video, timestep=args.timestep,
		endpoint_mode=args.endpoint_mode, collision_penalty=args.collision_penalty
	)
	
	if not args.nogui:
		s.render(ax=ax)
		plt.pause(0.01)

	# ACTIONS = [(200, 0)] #, (0, np.pi/6), (50, np.pi/12)]
	# Takes 20 steps to turn around
	# ACTIONS = [(100, 0.1)]
	ACTIONS = [(100, 0)] # TEMP: For LIDAR debugging
	rng = np.random.default_rng(42)
	car_actions = [rng.choice(ACTIONS) for _ in range(args.num_cars)]

	import time
	starttime = time.time()
	for i in range(80):
		obs, reward, done, info = s.step(car_actions)
		print('reward', reward)
		if not args.nogui:
			s.render(ax=ax)
			plt.pause(0.01)
	print('total time', time.time()-starttime)

	if not args.nogui:
		plt.pause(10)
	
	s.close()

	# FIXME: Some problems right off the bat:
	# - I can't keep the plt window open without pausing it for a long period of time.
