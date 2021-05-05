import numpy as np
from utils import *
from sim_map import Map
import matplotlib.pyplot as plt

import gym
from gym import spaces

# https://gist.github.com/danieljfarrell/faf7c4cafd683db13cbc
# public domain

PHI_MIN = -np.pi/2
PHI_MAX = np.pi/2 

V_MIN = -100
V_MAX = 100

DPHI_MIN = -5
DPHI_MAX = 5

TIMESTEP = 0.05

CAR_L = 5
CAR_LEN = 6
CAR_W = 3

LIDAR_MIN = -np.pi/2
LIDAR_MAX = np.pi/2
LIDAR_N = 10

DPHI_PENALTY_THRESHOLD = np.pi/200 # FIXME: May be too small or large?
DPHI_PENALTY_MAX = np.pi/40 # FIXME: May be too small or large?

N_NEARBY_CARS = 3

class Car:
    state = np.array([0,0,0,0]) #x, y, theta, phi
    velocity = np.array([0,0,0,0]) #dx, dy, theta, dphi
    collided = False

    def __init__(self, start_state, goal_state, width, height, is_autonomous=True):
        self.state = np.array(start_state)
        self.goal_state = np.array(goal_state) # x, y; theta and phi can be anything
        self.is_autonomous = is_autonomous # TODO: Implement human-driven car!
        self.width = width
        self.height = height
        
    def get_vertices(self, padding=0):
        x, y, t, _ = self.state
        R = rot_matrix(-t)
        pos_R = R @ np.array([x, y])
        # FIXME: Should top and bottom be flipped? It might not matter.
        tl = R.T @ (pos_R + np.array([-self.width / 2 - padding,  self.height / 2 + padding]))
        tr = R.T @ (pos_R + np.array([ self.width / 2 + padding,  self.height / 2 + padding]))
        bl = R.T @ (pos_R + np.array([-self.width / 2 - padding, -self.height / 2 - padding]))
        br = R.T @ (pos_R + np.array([ self.width / 2 + padding, -self.height / 2 - padding]))
        return [tl, tr, bl, br]
        
    def get_segments(self, padding=0):     
        [tl, tr, bl, br] = self.get_vertices(padding)
        return [[tl,bl],[bl,br],[tr,br],[tl,tr]]
    
    def intersect(self, other, padding=0):
        for seg1 in self.get_segments(padding):
            for seg2 in other.get_segments(padding):
                if(intersect_segments(seg1, seg2)):
                    return True
        return False

    #apply control inputs, changing velocity
    def control(self, v, dphi):
        v = min(max(v, V_MIN), V_MAX)
        dphi = min(max(dphi, DPHI_MIN), DPHI_MAX)

        # FIXME: This is also janky oh man
        mat1 = np.array([np.cos(np.pi - self.state[2]), np.sin(np.pi - self.state[2]), -np.tan(np.pi - self.state[3]) / CAR_L, 0])
        mat2 = np.array([0,0,0,1])
        self.velocity = v * mat1 + dphi * mat2
        # FIXME: This is so janky
        dx, dy, dtheta, dphi = self.velocity
        self.velocity = np.array([dy, dx, dtheta, dphi])

    #step the car, changing state as per velocity
    def step(self, action, timestep):
        if not self.collided:
            self.control(*action)
            self.state += timestep * self.velocity

    def distance_to_goal(self):
        x, y, _, _ = self.state
        x_g, y_g = self.goal_state
        return np.sqrt((x - x_g)**2 + (y - y_g)**2)

    def reached_goal(self, threshold=0.5):
        return self.distance_to_goal() <= threshold

    def collide(self):
        # Sets collided to true and stops the car and stuff
        self.collided = True
        self.velocity = np.zeros(4)

class Sim(gym.Env):
    def __init__(self, num_cars, map_img_path, path_reversal_probability=0, angle_min=-np.pi, angle_max=np.pi, save_video=True, timestep=0.1, spawn_padding=1, max_episode_steps=80):
        self.save_video = save_video
        self.timestep = timestep
        self.spawn_padding = spawn_padding
        self.max_episode_steps = max_episode_steps
        self.num_cars = num_cars

        self.map = Map(map_img_path, path_reversal_probability, angle_min, angle_max, LIDAR_MIN, LIDAR_MAX)

        self.reset()

    # TODO
    # @property
    # def state_space(self):
    #     obs_len_per_car =
    #     return spaces.Box(low=-4, high=max(self.map.img_shape), shape=(len(self.cars) * 3,), dtype=np.float32)

    # @property
    # def observation_space(self):
    #     return self.cars[0].observation_space

    # @property
    # def action_space(self):
    #     return self.cars[0].action_space

    @property
    def is_terminal(self):
        return len(self.cars) == 0 or self.time >= self.max_episode_steps

    def reset(self):
        self.time = 0
        self.cars = []
        i = 0
        while i < self.num_cars:
            # TODO: Possibly add the ability to add cars mid-simulation.
            start, end, start_angle = self.map.choose_path(padding=self.spawn_padding)
            if not self.check_collisions_with(*start, start_angle, padding=self.spawn_padding):
                i += 1
                self.spawn_car(*start, start_angle, *end)
            # else: print(i, 'collided')

        return self.get_obs()

    def spawn_car(self, x, y, theta, x_goal, y_goal):
        car = Car((x, y, theta, 0), (x_goal, y_goal), self.map.car_width, self.map.car_height)
        self.cars.append(car)

    def remove_car(self, index):
        del self.cars[index]

    def raycast(self, x, y, angle):
        best = float('inf')
        for car in self.cars:
            for segment in car.get_segments():
                d, _ = intersect_ray_segment([x,y], angle, segment[0], segment[1])
                if d != -1 and d < best:
                    best = d
        return best

    def lidar(self, car):
        x, y, angle, _ = car.state
        # ret = [self.raycast(x, y, t) for t in range(angle + LIDAR_MIN, angle + LIDAR_MAX, LIDAR_N-1)]
        # ret.append(self.raycast(x, y, angle + LIDAR_MAX))
        ret = [self.raycast(x, y, t) for t in np.linspace(LIDAR_MIN + angle, LIDAR_MAX + angle, LIDAR_N, endpoint=True)]
        return ret

    #returns a list of the other cars sorted by distance
    def nearby_cars(self, car, num_cars=None):
        ret = []
        for other in self.cars:
            if other is not car:
                ret.append(other)
        ret.sort(key=lambda c: (c.state[0] - car.state[0])**2 + (c.state[1] - car.state[1])**2)
        if num_cars is None:
            return ret
        return ret[:num_cars]

    def check_collisions(self):
        # Set car to collision mode
        for i in range(len(self.cars)):
            for j in range(i+1, len(self.cars)):
                if self.cars[i].collided and self.cars[j].collided:
                    pass
                else:
                    collided = False
                    for seg1 in self.cars[i].get_segments():
                        for seg2 in self.cars[j].get_segments():
                            if intersect_segments(seg1, seg2):
                                self.cars[i].collide()
                                self.cars[j].collide()
                                collided = True
                                break
                        if collided: break
        # Count number of cars caught in collision
        count = sum([car.collided for car in self.cars])
        return count

    def check_collisions_with(self, x, y, theta, padding=0):
        '''Checks for collisions with a car not yet added to the simulation. Has no side effects.'''
        car = Car((x, y, theta, 0), (0, 0), self.map.car_width, self.map.car_height)
        for i in range(len(self.cars)):
            for seg1 in car.get_segments(padding):
                for seg2 in self.cars[i].get_segments(padding):
                    if intersect_segments(seg1, seg2):
                        return True
        return False

    def render(self, ax=None):
        assert ax is not None
        self.map.render(self.cars, ax, save_frame=self.save_video)

    def get_obs(self):
        obs = []
        # Get observation (LIDAR, current velocity and pos, vel and pos of nearby cars)
        for car in self.cars:
            car_raycast = self.lidar(car)
            map_raycast = self.map.lidar(car, LIDAR_N)
            raycast = [min(c, m) for c, m in zip(car_raycast, map_raycast)]
            neighbors = self.nearby_cars(car, N_NEARBY_CARS)
            
            curr_obs = []
            curr_obs.extend(raycast)
            curr_obs.extend(car.state)
            for neighbor in neighbors:
                n_x, n_y, n_theta, n_phi = neighbor.state
                n_dx, n_dy, n_dtheta, n_dphi = neighbor.velocity
                curr_obs.extend([n_x, n_y, n_theta, n_dx, n_dy, n_dtheta])

            obs.extend(curr_obs)
        return obs

    def step(self, actions):
        '''actions: (v, dphi)'''
        obs, reward, done, info = [], 0, False, {}

        self.time += 1

        to_remove = []
        for i, (car, action) in enumerate(zip(self.cars, actions)):
            x, y, theta, phi = car.state
            v, dphi = action
            car.step(action, self.timestep)
            # Reward closeness to goal
            # The *30 and min(...,3) basically means to try to get within 10 pixels of the target
            reward += min(1/car.distance_to_goal()*30, 3)
            # Penalize map collisions
            if car.collided or self.map.car_has_boundary_collision(np.array((x, y)), theta):
                car.collide()
                reward -= 3
            # Penalize large rotational velocity
            if np.abs(dphi) <= DPHI_PENALTY_THRESHOLD:
                # FIXME: Tweak values and also function shape (right now it's a shrug)
                reward -= lerp(normalize_between(np.abs(dphi), DPHI_PENALTY_THRESHOLD, DPHI_PENALTY_MAX), 0, 1/200)
            # Once car reaches goal, prepare to remove from simulation
            if car.reached_goal():
                to_remove.insert(0, i)
        
        # Penalize collisions between cars
        num_car_collisions = self.check_collisions()
        reward -= num_car_collisions * 3

        # TODO: Maybe penalize time

        obs = self.get_obs()

        # Remove finished cars
        for i in to_remove:
            self.remove_car(i)

        # Check number of cars remaining
        done = len(self.cars) == 0

        return np.array(obs), reward, done, info

    def close(self):
        if self.save_video: # TEMP: Eventually move this into map probably
            self.map.close(self.timestep)
