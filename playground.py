import numpy as np
from sim import Sim, ManualCar
import matplotlib.pyplot as plt

NUM_CARS = 1

if __name__ == '__main__':
	plt.ion()
	fig, ax = plt.subplots(figsize=(8,8))
	fig.canvas.set_window_title('AV Swarm Simulator')
	plt.show()

	s = Sim(0, 'maps/task1a_moreborders.png', save_video=False)
	# s.add_manual_car(fig)
	# s.add_random_car(4)
	s.spawn_car(340, 580, np.pi/6, 0, 0)
	s.render(ax=ax)
	plt.pause(0.01)

	ACTIONS = [(100, 0)] # TEMP: For LIDAR debugging
	rng = np.random.default_rng(42)
	car_actions = [rng.choice(ACTIONS) for _ in range(NUM_CARS)]

	for i in range(10):
		s.step(car_actions)
		s.render(ax=ax)
		plt.pause(0.01)

	s.close()
