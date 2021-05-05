import numpy as np
from utils import *
from sim_map import Map
import matplotlib.pyplot as plt

# https://gist.github.com/danieljfarrell/faf7c4cafd683db13cbc
# public domain

PHI_MIN = -np.pi/2
PHI_MAX = np.pi/2 

V_MIN = -5
V_MAX = 5

DPHI_MIN = -5
DPHI_MAX = 5

TIMESTEP = 0.01

CAR_L = 5
CAR_LEN = 6
CAR_W = 3

LIDAR_MIN = -np.pi/2
LIDAR_MAX = np.pi/2
LIDAR_N = 10

DPHI_PENALTY_THRESHOLD = np.pi/12 # FIXME: May be too small or large?
DPHI_PENALTY_MAX = np.pi/4 # FIXME: May be too small or large?

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
        
    def get_vertices(self):
        x, y, t, _ = self.state
        R = rot_matrix(-t)
        pos_R = R @ np.array([x, y])
        tl = R.T @ (pos_R + np.array([-self.width / 2,  self.height / 2]))
        tr = R.T @ (pos_R + np.array([ self.width / 2,  self.height / 2]))
        bl = R.T @ (pos_R + np.array([-self.width / 2, -self.height / 2]))
        br = R.T @ (pos_R + np.array([ self.width / 2, -self.height / 2]))
        return [tl, tr, bl, br]
        
    def get_segments(self):     
        [tl, tr, bl, br] = self.get_vertices()
        return [[tl,bl],[bl,br],[tr,br],[tl,tr]]
    
    def intersect(self, other):
        for seg1 in self.get_segments():
            for seg2 in other.get_segments():
                if(intersect_segments(seg1, seg2)):
                    return True
        return False

    #apply control inputs, changing velocity
    def control(self, v, dphi):
        v = min(max(v, V_MIN), V_MAX)
        dphi = min(max(dphi, DPHI_MIN), DPHI_MAX)

        mat1 = np.array([np.cos(self.state[2]), np.sin(self.state[2]), np.tan(self.state[3]) / CAR_L, 0])
        mat2 = np.array([0,0,0,1])
        self.velocity = v * mat1 + dphi * mat2

    #step the car, changing state as per velocity
    def step(self, action):
        if not self.collided:
            self.control(*action)
            self.state += TIMESTEP * self.velocity

    def distance_to_goal(self):
        x, y, _, _ = self.state
        x_g, y_g = self.goal_state
        return np.sqrt((x - x_g)**2 + (y - y_g)**2)

    def reached_goal(self, threshold=0.5):
        return self.distance_to_goal() <= threshold

class Sim:
    def __init__(self, num_cars, map_img_path, path_reversal_probability=0, angle_min=-np.pi, angle_max=np.pi, save_video=True):
        self.save_video = save_video
        self.cars = []
        self.map = Map(map_img_path, path_reversal_probability, angle_min, angle_max, LIDAR_MIN, LIDAR_MAX)
        i = 0
        while i < num_cars:
            # TODO: Possibly add the ability to add cars mid-simulation.
            start, end, start_angle = self.map.choose_path()
            if not self.check_collisions_with(*start, start_angle):
                i += 1
                self.spawn_car(*start, start_angle, *end)
            else: print(i, 'collided')

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
        count = 0
        for i in range(len(self.cars)):
            for j in range(i+1, len(self.cars)):
                for seg1 in self.cars[i].get_segments():
                    for seg2 in self.cars[j].get_segments():
                        if intersect_segments(seg1, seg2):
                            self.cars[i].collided = True
                            self.cars[j].collided = True
                            count += 1
        return count

    def check_collisions_with(self, x, y, theta):
        '''Checks for collisions with a car not yet added to the simulation. Has no side effects.'''
        car = Car((x, y, theta, 0), (0, 0), self.map.car_width, self.map.car_height)
        for i in range(len(self.cars)):
            for seg1 in car.get_segments():
                for seg2 in self.cars[i].get_segments():
                    if intersect_segments(seg1, seg2):
                        return True
        return False

    def render(self):
        self.map.render(self.cars, save_frame=self.save_video)

    def step(self, actions):
        '''actions: (v, dphi)'''
        obs, reward, done, info = [], 0, False, {}

        to_remove = []
        for i, (car, action) in enumerate(zip(self.cars, actions)):
            x, y, theta, phi = car.state
            v, dphi = action
            car.step(action)
            # Reward closeness to goal
            reward += min(car.distance_to_goal(), 3)
            # Penalize map collisions
            if self.map.car_has_boundary_collision(np.array((x, y)), theta):
                car.collided = True
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

            obs.append(curr_obs)

        # Remove finished cars
        for i in to_remove:
            self.remove_car(i)

        # Check number of cars remaining
        done = len(self.cars) == 0

        return obs, reward, done, info

    def close(self):
        if self.save_video: # TEMP: Eventually move this into map probably
            self.map.close()
