import numpy as np
import cv2

rng = np.random.default_rng(42)

class Map:
	def __init__(self, img_path, path_reversal_probability = 0, angle_min = 0, angle_max = 2 * np.pi):
		self.img_path = img_path
		self.img = cv2.imread(img_path)
		self.path_reversal_probability = path_reversal_probability
		self.angle_min = angle_min
		self.angle_max = angle_max

		# Find all red, blue, and black pixels (OpenCV uses BGR)
		img_b = self.img[:,:,0]
		img_g = self.img[:,:,1]
		img_r = self.img[:,:,2]

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
		if self.has_boundary_collision(start, start_angle):
			return self.choose_path()
		return start, end, start_angle

	def has_boundary_collision(self, point, angle):
		# Rotate the image and the point
		R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
		boundary_R = (R @ self.boundary_points.T).T
		point_R = R @ point
		# Check the vectors between all boundary points and the point
		collisions = np.where(np.logical_and(
			np.abs(boundary_R[:,0] - point_R[0]) <= self.car_width / 2,
			np.abs(boundary_R[:,1] - point_R[1]) <= self.car_height / 2
		))
		return len(collisions[0]) > 0

	def scan(self, car_idx, car_poses):
		pass

	def render(self, car_poses):
		pass
