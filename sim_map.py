import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from utils import *

rng = np.random.default_rng(42)

class Map:
	def __init__(self, img_path, path_reversal_probability = 0, angle_min = 0, angle_max = 2 * np.pi):
		self.img_path = img_path
		self.img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
		self.path_reversal_probability = path_reversal_probability
		self.angle_min = angle_min
		self.angle_max = angle_max

		self.img_shape = self.img.shape[:2]

		# Find all red, blue, and black pixels
		img_r = self.img[:,:,0]
		img_g = self.img[:,:,1]
		img_b = self.img[:,:,2]

		# FIXME: These points might be iffy because right now, the origin is the top left, and it... increases as it goes down and right? Odd.
		# Right now, I'm flipping it so that the origin is the bottom left and it increases as it goes up and right; not sure if that fixes it.

		start_points = np.where(np.logical_and(img_r < 250, img_g < 250, img_b == 255))
		self.start_points = np.zeros((len(start_points[0]), 2))
		self.start_points[:,0] = start_points[1] # col = x
		self.start_points[:,1] = (self.img.shape[0] - 1) - start_points[0] # row = y

		end_points = np.where(np.logical_and(img_r == 255, img_g < 250, img_b < 250))
		self.end_points = np.zeros((len(end_points[0]), 2))
		self.end_points[:,0] = end_points[1] # col = x
		self.end_points[:,1] = (self.img.shape[0] - 1) - end_points[0] # row = y

		boundary_points = np.where(np.logical_and(img_r == 0, img_g == 0, img_b == 0))
		self.boundary_points = np.zeros((len(boundary_points[0]), 2))
		self.boundary_points[:,0] = boundary_points[1] # col = x
		self.boundary_points[:,1] = (self.img.shape[0] - 1) - boundary_points[0] # row = y

		# Determine car size for rendering and scan
		car_size_points = np.where(np.logical_and(img_r < 250, img_g == 255, img_b < 250))
		self.car_width = np.max(car_size_points[1]) - np.min(car_size_points[1]) + 1
		self.car_height = np.max(car_size_points[0]) - np.min(car_size_points[0]) + 1

		# Create window for rendering
		plt.ion()
		self.fig, self.ax = plt.subplots(figsize=(8,8))
		self.fig.canvas.set_window_title('AV Swarm Simulator')
		plt.show()

	def get_car_size(self):
		return self.car_width, self.car_height

	def choose_path(self):
		'''Finds a starting and ending point for a car. Returns each as (row, col) indices.'''
		# Randomly select a start point
		start_idx = rng.randint(self.start_points.shape[0])
		start = self.start_points[start_idx,:]
		# Randomly select an end point
		end_idx = rng.randint(self.end_points.shape[0])
		end = self.end_points[end_idx,:]
		# Possibly randomly reverse if allowed
		if rng.uniform() < self.path_reversal_probability:
			end_temp = end
			end = start
			start = end_temp
		# Randomly choose an angle
		start_angle = rng.uniform(low=self.angle_min, high=self.angle_max)
		# Ensure no collisions
		if self.car_has_boundary_collision(start, start_angle):
			return self.choose_path()
		return start, end, start_angle

	def car_has_boundary_collision(self, point, angle):
		# Rotate the image and the point
		R = rot_matrix(angle)
		boundary_R = (R @ self.boundary_points.T).T
		point_R = R @ point
		# Check the vectors between all boundary points and the point
		collisions = np.where(np.logical_and(
			np.abs(boundary_R[:,0] - point_R[0]) <= self.car_width / 2,
			np.abs(boundary_R[:,1] - point_R[1]) <= self.car_height / 2
		))
		return len(collisions[0]) > 0
		# FIXME: On second thought, will this miss things that are in the box of a
		# boundary pixel but aren't exactly on that pixel's corner?

	def raycast(self, x, y, angle):
		# Rotate image into these coordinates for purely horizontal ray traversal
		R = rot_matrix(-angle)
		pos_R = R @ np.array([x, y])
		
		# Form boundaries of image so they can also be rotated
		height, width = self.img_shape
		bl_R = R @ np.array([0, 0])
		br_R = R @ np.array([width, 0])
		tl_R = R @ np.array([0, height])
		tr_R = R @ np.array([width, height])

		# Find bounds of iteration and ray length for the only one it collides with
		map_segments = [(bl_R, br_R), (br_R, tr_R), (tl_R, tr_R), (bl_R, tl_R)]
		map_end = None
		for seg in map_segments:
			t1, map_end = intersect_ray_segment(pos_R, 0, *seg)
			if t1 != -1:
				break
		map_end_x = map_end[0]

		# Walk straight to the right with a step size of one pixel-ish (should I do half a pixel?)
		x = pos_R[0]
		while x <= map_end_x:
			# Get current coord in original coordinates and floor to bottom left
			curr_pos_R = np.array([x, pos_R[1]])
			curr_pos_locked = np.floor(R.T @ curr_pos_R)
			matches = np.where(np.logical_and(
				curr_pos_locked[0] == self.boundary_points[:,0],
				curr_pos_locked[1] == self.boundary_points[:,1]
			))
			# Check if boundary points contains that point
			if len(matches[0]) > 0:
				break
			x += 0.5

		# Get ray length from stopping point
		x = min(x, map_end_x)
		raylength = x - pos_R[0]
		return raylength

	def lidar(self, car, n_rays):
		x, y, angle, _ = car.state
		ret = [self.raycast(x, y, t) for t in np.linspace(self.angle_min + angle, self.angle_max + angle, n_rays, endpoint=True)]
		return ret

	def render(self, car_poses, pause_length=0.001):
		self.ax.clear()
		self.ax.imshow(self.img)
		for x, y, angle in car_poses:
			self.ax.add_patch(Rectangle(
				(x, y),
				self.car_width,
				self.car_height,
				angle=angle,
				edgecolor='black',
				facecolor='white',
				fill=True,
				lw=2
			))
		self.ax.axis('off')
		plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
		# FIXME: I probably shouldn't be pausing in this function,
		# but not sure where else to do it. Let me know if you have thoughts.
		plt.pause(pause_length)
