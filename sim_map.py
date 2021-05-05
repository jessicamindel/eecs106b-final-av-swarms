import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import matplotrecorder
from utils import *

rng = np.random.default_rng(42)

class Map:
	def __init__(self,
		img_path, path_reversal_probability=0.0,
		angle_min=0.0, angle_max=2*np.pi,
		angle_mode='auto', angle_noise=0.0,
		lidar_angle_min=-np.pi, lidar_angle_max=np.pi,
	):
		self.img_path = img_path
		self.img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
		self.path_reversal_probability = path_reversal_probability
		self.angle_min = angle_min
		self.angle_max = angle_max
		self.angle_mode = angle_mode
		self.angle_noise = angle_noise
		self.lidar_angle_min = lidar_angle_min
		self.lidar_angle_max = lidar_angle_max

		self.img_shape = self.img.shape[:2]

		# Find all red, blue, and black pixels
		img_r = self.img[:,:,0]
		img_g = self.img[:,:,1]
		img_b = self.img[:,:,2]

		# FIXME: These points might be iffy because right now, the origin is the top left, and it... increases as it goes down and right? Odd.
		# Right now, I'm flipping it so that the origin is the bottom left and it increases as it goes up and right; not sure if that fixes it.

		# Start direction points are used to calculate the starting angle; they form a line that bounds the start section pointing in the direction
		# that cars in that section should go. In the future, we could add a similar feature for the end point, but we currently don't include theta
		# in the goal state.
		start_dir_points = np.where(np.logical_and(img_r == 0, np.logical_and(img_g == 0, img_b == 255)))
		self.start_dir_points = np.ones((len(start_dir_points[0]), 2))
		self.start_dir_points[:,0] = start_dir_points[1] # col = x
		self.start_dir_points[:,1] = start_dir_points[0] # row = y

		start_points = np.where(np.logical_and(np.logical_and(img_r < 250, img_r > 0), np.logical_and(np.logical_and(img_g < 250, img_g > 0), img_b == 255)))
		self.start_points = np.zeros((len(start_points[0]), 2))
		self.start_points[:,0] = start_points[1] # col = x
		self.start_points[:,1] = start_points[0] # row = y

		end_points = np.where(np.logical_and(img_r == 255, np.logical_and(img_g < 250, img_b < 250)))
		self.end_points = np.zeros((len(end_points[0]), 2))
		self.end_points[:,0] = end_points[1] # col = x
		self.end_points[:,1] = end_points[0] # row = y

		boundary_points = np.where(np.logical_and(img_r == 0, np.logical_and(img_g == 0, img_b == 0)))
		self.boundary_points = np.zeros((len(boundary_points[0]), 2))
		self.boundary_points[:,0] = boundary_points[1] # col = x
		self.boundary_points[:,1] = boundary_points[0] # row = y

		# Determine car size for rendering and scan
		car_size_points = np.where(np.logical_and(img_r < 250, img_g == 255, img_b < 250))
		self.car_width = np.max(car_size_points[1]) - np.min(car_size_points[1]) + 1
		self.car_height = np.max(car_size_points[0]) - np.min(car_size_points[0]) + 1

		self.user_text_point = (min(car_size_points[1]), min(car_size_points[0]))

	def get_car_size(self):
		return self.car_width, self.car_height

	def choose_path(self, padding=0):
		'''Finds a starting and ending point for a car. Returns each as (row, col) indices.'''
		# Randomly select a start point
		start_idx = rng.integers(self.start_points.shape[0])
		start = self.start_points[start_idx,:]
		# Randomly select an end point
		end_idx = rng.integers(self.end_points.shape[0])
		end = self.end_points[end_idx,:]
		# Possibly randomly reverse if allowed
		if rng.uniform() < self.path_reversal_probability:
			end_temp = end
			end = start
			start = end_temp
		if self.angle_mode[:4] == 'auto':
			# Find the nearest point along the direction line
			nearest_dir_idx = np.argmin(np.linalg.norm(self.start_dir_points - start, axis=1))
			dir_vec = start - self.start_dir_points[nearest_dir_idx]
			# Determine the angle using the resulting vector
			start_angle = np.mod(-np.arctan2(*dir_vec), 2*np.pi)
			if self.angle_mode == 'auto_noise':
				# FIXME: 0.6 may be too small of a variance, but I tried to avoid sizes that often yielded samples > 1.
				start_angle += self.angle_noise * rng.normal(0, 0.6)
		elif self.angle_mode == 'random':
			# Randomly choose an angle
			start_angle = rng.uniform(low=self.angle_min, high=self.angle_max)
		else:
			raise ValueError(f'In Map.choose_path, angle_mode must be either \'auto\', \'auto_noise\', or \'random\'; {self.angle_mode} is not a valid mode.')
		# Ensure no collisions
		if self.car_has_boundary_collision(start, start_angle, padding):
			return self.choose_path(padding)
		return start, end, start_angle

	def car_has_boundary_collision(self, point, angle, padding=0):
		# Rotate the image and the point
		R = rot_matrix(angle)
		boundary_R = (R @ self.boundary_points.T).T
		point_R = R @ point
		# Check the vectors between all boundary points and the point
		collisions = np.where(np.logical_and(
			np.abs(boundary_R[:,0] - point_R[0]) <= self.car_width / 2 + padding,
			np.abs(boundary_R[:,1] - point_R[1]) <= self.car_height / 2 + padding
		))
		return len(collisions[0]) > 0
		# FIXME: On second thought, will this miss things that are in the box of a
		# boundary pixel but aren't exactly on that pixel's corner?

	# FIXME: It's possible I have the y direction wrong. This will need a revisit.
	def raycast(self, x, y, angle):
		# Rotate image into these coordinates for purely horizontal ray traversal
		R = rot_matrix(-angle)
		pos_R = R @ np.array([x, y])
		
		# Form boundaries of image so they can also be rotated
		height, width = self.img_shape
		bl_R = R @ np.array([0, height])
		br_R = R @ np.array([width, height])
		tl_R = R @ np.array([0, 0])
		tr_R = R @ np.array([width, 0])

		# Find bounds of iteration and ray length for the only one it collides with
		map_segments = [(bl_R, br_R), (br_R, tr_R), (tl_R, tr_R), (bl_R, tl_R)]
		map_end_R = None
		for seg in map_segments:
			t1, map_end_R = intersect_ray_segment(pos_R, 0, *seg)
			if t1 != -1:
				break
		map_end_y_R = map_end_R[1]

		# # Walk straight up with a step size of one pixel-ish (should I do half a pixel?)
		# y = pos_R[1]
		# collided = False
		# while y >= map_end_y:
		# 	# Get current coord in original coordinates and floor to bottom left
		# 	curr_pos_R = np.array([pos_R[0], y])
		# 	curr_pos_locked = np.floor(R.T @ curr_pos_R)
		# 	matches = np.where(np.logical_and(
		# 		curr_pos_locked[0] == self.boundary_points[:,0],
		# 		curr_pos_locked[1] == self.boundary_points[:,1]
		# 	))
		# 	# Check if boundary points contains that point
		# 	if len(matches[0]) > 0:
		# 		collided = True
		# 		break
		# 	y -= 0.5

		# # Get ray length from stopping point
		# y = max(y, map_end_y)
		# raylength = y - pos_R[1] if collided else float('inf')
		# return raylength
		
		# Create grid-aligned list of pixels over which to iterate
		ys_R = np.arange(pos_R[1], map_end_y_R, 0.5)
		steps_R = np.zeros((ys_R.shape[0], 2))
		steps_R[:,0] = pos_R[0]
		steps_R[:,1] = ys_R
		steps_locked = np.floor((R.T @ steps_R.T).T).astype(int)
		
		# Find the nearest boundary pixel
		pixels_r = self.img[steps_locked[:,0], steps_locked[:,1], 0]
		pixels_g = self.img[steps_locked[:,0], steps_locked[:,1], 1]
		pixels_b = self.img[steps_locked[:,0], steps_locked[:,1], 2]
		boundary_point_idxs = np.where(np.logical_and(pixels_r == 0, np.logical_and(pixels_g == 0, pixels_b == 0)))[0]

		# If none was found, there is no bound on length
		if len(boundary_point_idxs) == 0:
			return float('inf')

		# Otherwise, find the distance between the car and this pixel
		boundary_point_R = steps_R[boundary_point_idxs[0]]
		return pos_R[1] - boundary_point_R[1]

	def lidar(self, car, n_rays):
		x, y, angle, _ = car.state
		ret = [self.raycast(x, y, t) for t in np.linspace(self.lidar_angle_min + angle, self.lidar_angle_max + angle, n_rays, endpoint=True)]
		return ret

	def draw_car(self, ax, x, y, angle, center='point', text=None, front_color='green'):
		FRONT_RATIO = 0.1
		# Find center point
		pos = np.array([x, y])
		R = rot_matrix(-angle)
		pos_R = R @ pos
		top_right = R.T @ (pos_R - np.array([self.car_width / 2, self.car_height / 2]))
		top_middle_short = R.T @ (pos_R - np.array([0, self.car_height / 4]))
		ax.add_patch(Rectangle(
			top_right,
			self.car_width,
			self.car_height,
			angle=angle*180/np.pi,
			edgecolor='black',
			facecolor='white',
			fill=True,
			lw=2
		))
		ax.add_patch(Rectangle(
			top_right,
			self.car_width,
			self.car_height*FRONT_RATIO,
			angle=angle*180/np.pi,
			edgecolor=front_color,
			facecolor=front_color,
			fill=True,
			lw=2
		))
		if center == 'point':
			plt.plot(x, y, 'ko', markersize=4)
		elif center == 'vector': # FIXME: Doesn't work :(
			plt.plot(x, y, top_middle_short[0] - x, top_middle_short[1] - y)
		elif center == 'text' and text is not None:
			plt.text(x, y, text, rotation=-angle*180/np.pi, fontsize=6, ha='center')

	def render(self, cars, non_rl_cars, ax, save_frame=True):
		ax.clear()
		ax.imshow(self.img)
		for i, car in enumerate(cars):
			x, y, angle, _ = car.state
			self.draw_car(ax, x, y, angle, center='text', text=str(i))
		for i, car in enumerate(non_rl_cars):
			x, y, angle, _ = car.state
			classname = type(car).__name__
			if classname == 'ManualCar':
				self.draw_car(ax, x, y, angle, center='text', text=f'M{i}', front_color='blue')
				# Draw goal point
				plt.plot(*car.goal_state, 'bo', markersize=10)
				plt.text(*car.goal_state, f'M{i}', c='white', fontsize=6, ha='center', va='center')
				# TODO: Draw velocity information
				plt.text(*self.user_text_point, f'Step size:\nv: {car.v_curr}\ndphi: {car.dphi_curr}', fontsize=6, ha='left', va='top')
			if classname == 'RandomCar':
				self.draw_car(ax, x, y, angle, center='text', text=f'R{i}', front_color='red')

		ax.axis('off')
		plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
		if save_frame: matplotrecorder.save_frame()

	def close(self, timestep):
		matplotrecorder.save_movie(f'img/av_swarm_{int(time.time())}.mp4', timestep)
