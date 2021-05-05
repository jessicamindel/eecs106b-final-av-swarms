import numpy as np
import matplotlib.pyplot as plt
import itertools

from utils import *
from sim_map import Map

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

V_MANUAL_INCREMENT = 5.0
DPHI_MANUAL_INCREMENT = 0.001

TIMESTEP = 0.05

CAR_L = 5
CAR_LEN = 6
CAR_W = 3
CAR_COLLIDER_BOUND2 = CAR_LEN * CAR_LEN + CAR_W * CAR_W

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

    def __init__(self, start_state, goal_state, width, height, is_rational=True):
        self.state = np.array(start_state)
        self.goal_state = np.array(goal_state) # x, y; theta and phi can be anything
        self.is_rational = is_rational # TODO: Implement human-driven car!
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
        dx = self.state[0] - other.state[0]
        dy = self.state[1] - ohter.state[1]
        if dx*dx + dy*dy < CAR_COLLIDER_BOUND2:
            return False
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

    def step(self, action, timestep):
        if not self.collided:
            self.control(*action)
            self.state += timestep * self.velocity
            return action
        return (0, 0)

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

class ManualCar(Car):
    def __init__(self,
        start_state, goal_state, width, height, figure,
        key_forward='up', key_back='down', key_left='left', key_right='right',
        key_v_up='j', key_v_down='h', key_dphi_up='u', key_dphi_down='y'
    ):
        super().__init__(start_state, goal_state, width, height, is_rational=False)
        self.figure = figure

        self.pressed_keys = dict()
        self.pressed_keys[key_forward] = False
        self.pressed_keys[key_back] = False
        self.pressed_keys[key_left] = False
        self.pressed_keys[key_right] = False
        self.pressed_keys[key_v_up] = False
        self.pressed_keys[key_v_down] = False
        self.pressed_keys[key_dphi_up] = False
        self.pressed_keys[key_dphi_down] = False

        self.v_curr = 10.0
        self.dphi_curr = 0.001

        # Source for anonymous objects: https://stackoverflow.com/a/29480317/3843659
        self.keys = type('',(object,),{
            "forward": key_forward,
            "back": key_back,
            "left": key_left,
            "right": key_right,
            "v": type('',(object,),{
                "up": key_v_up,
                "down": key_v_down
            }),
            "dphi": type('',(object,),{
                "up": key_dphi_up,
                "down": key_dphi_down
            })
        })

        # Set up keypress events
        def onkeypress(event):
            if event.key in self.pressed_keys:
                self.pressed_keys[event.key] = True

        def onkeyrelease(event):
            if event.key in self.pressed_keys:
                self.pressed_keys[event.key] = False

        self.cid_keypress = figure.canvas.mpl_connect('key_press_event', onkeypress)
        self.cid_keyrelease = figure.canvas.mpl_connect('key_release_event', onkeyrelease)

    def step(self, timestep):
        # Increase and decrease velocity
        if self.pressed_keys[self.keys.v.up]:
            self.v_curr += V_MANUAL_INCREMENT
        if self.pressed_keys[self.keys.v.down]:
            self.v_curr -= V_MANUAL_INCREMENT
        # Increase and decrease angular velocity
        if self.pressed_keys[self.keys.dphi.up]:
            self.dphi_curr += DPHI_MANUAL_INCREMENT
        if self.pressed_keys[self.keys.dphi.down]:
            self.dphi_curr -= DPHI_MANUAL_INCREMENT
        
        # If no keys are pressed, no action should be taken
        v = 0.0
        dphi = 0.0
        # Move forward and back
        if self.pressed_keys[self.keys.forward]:
            v += self.v_curr
        if self.pressed_keys[self.keys.back]:
            v -= self.v_curr
        # Turn left and right
        if self.pressed_keys[self.keys.right]:
            dphi += self.dphi_curr
        if self.pressed_keys[self.keys.left]:
            dphi -= self.dphi_curr
        
        return super().step((v, dphi), timestep)

    def __del__(self):
        self.figure.canvas.mpl_disconnect(self.cid_keypress)
        self.figure.canvas.mpl_disconnect(self.cid_keyrelease)

class RandomCar(Car):
    def __init__(self, start_state, goal_state, width, height):
        super().__init__(start_state, goal_state, width, height, is_rational=False)

    def step(self, timestep):
        # TODO: Generate a random action within some bounds
        return super().step(None, timestep)

class Sim(gym.Env):
    def __init__(self,
        num_cars, map_img_path, path_reversal_probability=0,
        angle_min=-np.pi, angle_max=np.pi, spawn_padding=1,
        angle_mode='auto', angle_noise=0.0,
        save_video=True, timestep=0.1, max_episode_steps=80
    ):
        self.save_video = save_video
        self.timestep = timestep
        self.spawn_padding = spawn_padding
        self.max_episode_steps = max_episode_steps
        self.num_cars = num_cars

        self.map = Map(
            map_img_path, path_reversal_probability,
            angle_min, angle_max,
            angle_mode, angle_noise,
            LIDAR_MIN, LIDAR_MAX
        )

        self.actual_n_nearby_cars = min(N_NEARBY_CARS, self.num_cars-1)

        self.NEIGHBOR_OBS_LEN = 6
        self.LOCAL_OBS_LEN = LIDAR_N + 4
        self.OBS_LEN_PER_CAR = self.actual_n_nearby_cars*self.NEIGHBOR_OBS_LEN + self.LOCAL_OBS_LEN
        
        # I'm not sure what these are. The values are guesses
        self.dim_local_o = self.LOCAL_OBS_LEN
        self.dim_flat_o = self.dim_local_o
        # self.dim_local_o = N_NEARBY_CARS
        # self.dim_flat_o = self.dim_local_o
        self.dim_rec_o = (self.actual_n_nearby_cars, self.NEIGHBOR_OBS_LEN)
        self.dim_mean_embs = (self.actual_n_nearby_cars, self.NEIGHBOR_OBS_LEN)
        self.dim_o = np.prod(self.dim_rec_o) + self.dim_local_o
 
        self.reset()

    @property
    def state_space(self):
        return spaces.Box(low=-max(self.map.img_shape), high=max(self.map.img_shape), shape=(self.dim_o,), dtype=np.float32)

    @property
    def observation_space(self):
        # Observation space of one car
        ob_space = spaces.Box(low=-max(self.map.img_shape), high=max(self.map.img_shape), shape=(self.dim_o,), dtype=np.float32)
        ob_space.dim_local_o = self.dim_local_o
        ob_space.dim_flat_o = self.dim_flat_o
        ob_space.dim_rec_o = self.dim_rec_o
        ob_space.dim_mean_embs = self.dim_mean_embs
        return ob_space

    @property
    def action_space(self):
        # Actino space of one car
        return spaces.Box(low=np.array([V_MIN, PHI_MIN]), high=np.array([V_MAX, PHI_MAX]), dtype=np.float32)

    @property
    def is_terminal(self):
        return len(self.agents) == 0 or self.time >= self.max_episode_steps

    def reset(self):
        self.time = 0
        self.non_rl_cars = []
        self.agents = []
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
        self.agents.append(car)

    def add_manual_car(self, figure,
        key_forward='up', key_back='down', key_left='left', key_right='right',
        key_v_up='j', key_v_down='h', key_dphi_up='u', key_dphi_down='y'
    ):
        while True:
            start, end, start_angle = self.map.choose_path(padding=self.spawn_padding)
            if not self.check_collisions_with(*start, start_angle, padding=self.spawn_padding):
                break
        car = ManualCar(
            (*start, start_angle, 0), end, self.map.car_width, self.map.car_height, figure,
            key_forward, key_back, key_left, key_right, key_v_up, key_v_down, key_dphi_up, key_dphi_down
        )
        self.non_rl_cars.append(car)

    def remove_car(self, index, non_rl=False):
        if non_rl:
            del self.non_rl_cars[index]
        else:
            del self.agents[index]

    def raycast(self, x, y, angle):
    def raycast(self, x, y, angle, exclude = None):
        best = float('inf')
        for car in itertools.chain(self.agents, self.non_rl_cars):
            if car is exclude:
                continue
            for segment in car.get_segments():
                d, _ = intersect_ray_segment([x,y], angle, segment[0], segment[1])
                if d != -1 and d < best:
                    best = d
        return best

    def lidar(self, car):
        x, y, angle, _ = car.state
        angle -= np.pi/2
        ret = [self.raycast(x, y, t, car) for t in np.linspace(LIDAR_MIN + angle, LIDAR_MAX + angle, LIDAR_N, endpoint=True)]
        return ret

    #returns a list of the other cars sorted by distance
    def nearby_cars(self, car, num_cars=None):
        ret = []
        for other in itertools.chain(self.agents, self.non_rl_cars):
            if other is not car:
                ret.append(other)
        ret.sort(key=lambda c: (c.state[0] - car.state[0])**2 + (c.state[1] - car.state[1])**2)
        if num_cars is None:
            return ret
        return ret[:num_cars]

    def check_collisions(self):
        all_cars = self.agents + self.non_rl_cars
        # Set car to collision mode
        for i in range(len(all_cars)):
            for j in range(i+1, len(all_cars)):
                if all_cars[i].collided and all_cars[j].collided:
                    pass
                else:
                    collided = False
                    for seg1 in all_cars[i].get_segments():
                        for seg2 in all_cars[j].get_segments():
                            if intersect_segments(seg1, seg2):
                                all_cars[i].collide()
                                all_cars[j].collide()
                                collided = True
                                break
                        if collided: break
        # Count number of cars caught in collision
        count = sum([car.collided for car in self.agents])
        return count

    def check_collisions_with(self, x, y, theta, padding=0):
        '''Checks for collisions with a car not yet added to the simulation. Has no side effects.'''
        new_car = Car((x, y, theta, 0), (0, 0), self.map.car_width, self.map.car_height)
        for car in itertools.chain(self.agents, self.non_rl_cars):
            for seg1 in new_car.get_segments(padding):
                for seg2 in car.get_segments(padding):
                    if intersect_segments(seg1, seg2):
                        return True
        return False

    def render(self, ax=None):
        assert ax is not None
        self.map.render(self.agents, self.non_rl_cars, ax, save_frame=self.save_video)

    def get_obs(self):
        obs = []
        # Get observation (LIDAR, current velocity and pos, vel and pos of nearby cars)
        for car in self.agents:
            car_raycast = self.lidar(car)
            map_raycast = self.map.lidar(car, LIDAR_N)
            raycast = [min(c, m) for c, m in zip(car_raycast, map_raycast)]
            neighbors = self.nearby_cars(car, N_NEARBY_CARS)
            
            curr_obs = []
            # Neighbor stuff first
            for neighbor in neighbors:
                n_x, n_y, n_theta, n_phi = neighbor.state
                n_dx, n_dy, n_dtheta, n_dphi = neighbor.velocity
                curr_obs.extend([n_x, n_y, n_theta, n_dx, n_dy, n_dtheta])
            # Then stuff about the local observation
            curr_obs.extend(raycast)
            curr_obs.extend(car.state)

            obs.append(curr_obs)
        return np.array(obs)

    def get_per_car_reward(self, car, action):
        x, y, theta, phi = car.state
        v, dphi = action
        reward = 0
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
        return reward

    def step(self, actions):
        '''actions: (v, dphi)'''
        obs, reward, done, info = [], 0, False, {}

        self.time += 1

        to_remove = []
        for i, (car, action) in enumerate(zip(self.agents, actions)):
            v, dphi = action
            car.step(action, self.timestep)
            reward += self.get_per_car_reward(car, action)
            # Once car reaches goal, prepare to remove from simulation
            if car.reached_goal():
                to_remove.insert(0, i)

        to_remove_non_rl = []
        for i, car in enumerate(self.non_rl_cars):
            action = car.step(self.timestep)
            reward += self.get_per_car_reward(car, action)
            # Once car reaches goal, prepare to remove from simulation
            if car.reached_goal():
                to_remove_non_rl.insert(0, i)
        
        # Penalize collisions between cars
        num_car_collisions = self.check_collisions()
        reward -= num_car_collisions * 3

        # TODO: Maybe penalize time

        obs = self.get_obs()

        # Remove finished cars
        for i in to_remove:
            self.remove_car(i)
        for i in to_remove_non_rl:
            self.remove_car(i, non_rl=True)

        # Check number of cars remaining
        done = len(self.agents) == 0 and len(self.non_rl_cars) == 0

        return obs, reward, done, info

    def close(self):
        if self.save_video: # TEMP: Eventually move this into map probably
            self.map.close(self.timestep)
