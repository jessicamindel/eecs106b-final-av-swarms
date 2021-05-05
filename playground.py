import numpy as np
from sim import Sim, ManualCar
import matplotlib.pyplot as plt

if __name__ == '__main__':
	plt.ion()
	fig, ax = plt.subplots(figsize=(8,8))
	fig.canvas.set_window_title('AV Swarm Simulator')
	plt.show()

	s = Sim(0, 'maps/task1a.png', save_video=False)
	s.add_manual_car(fig)
	s.render(ax=ax)
	plt.pause(0.01)

	for i in range(1000):
		s.step([])
		s.render(ax=ax)
		plt.pause(0.01)

	s.close()
