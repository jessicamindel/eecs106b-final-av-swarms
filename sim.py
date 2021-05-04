import numpy as np
from utils import *

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

class Car:
    state = np.array([0,0,0,0]) #x, y, theta, phi
    velocity = np.array([0,0,0,0]) #dx, dy, theta, dphi
    collided = False


    def get_segments(self):
        x = self.state[0]
        y = self.state[1]
        t = self.state[2]
        dx = np.cos(t) * CAR_W / 2
        dy = np.sin(t) * CAR_LEN / 2

        p1 = np.array([x+dx, y+dy])
        p2 = np.array([x+dx, y-dy])
        p3 = np.array([x-dx, y-dy])
        p4 = np.array([x-dx, y+dy])
        return [[p1,p2],[p2,p3],[p3,p4],[p4, p1]]
    
    def intersect(self, other):
        for seg1 in self.get_segments():
            for seg2 in other.get_segments():
                if(intersect_segments(seg1, seg2)):
                    return True
        return False

    #apply control inputs, changing velocity
    def control(self, v, dphi):
        if v < V_MIN:
            v = V_MIN
        if v > V_MAX:
            v = V_MAX
        if dphi < DPHI_MIN:
            dphi = DPHI_MIN
        if dphi > DPI_MAX:
            dphi = DPHI_MAX

        mat1 = np.array([np.cos(self.state[2]), np.sin(self.state[2]), np.tan(self.state[3]) / CAR_L, 0])
        mat2 = np.array([0,0,0,1])
        self.velocity = v * mat1 + dphi * mat2

    #step the car, changing state as per velocity
    def step(self):
        if not self.collided:
            self.state += TIMESTEP * self.velocity

class Sim:
    cars = []

    def spawn_car(self, x, y, theta):
        car = Car()
        car.state[0] = x
        car.state[1] = y
        car.state[2] = theta
        self.cars.push(car)

    def remove_car(index):
        del car[index]

    def raycast(self, x, y, angle):
        best = float('inf')
        for car in self.cars:
            for segment in car.get_segments():
                d = intersect_ray_segment([x,y], angle, segment[0], segment[1])
                if d != -1 and d < best:
                    best = d
        return best

    def lidar(self, car):
        x = car.state[0]
        y = car.state[1]
        angle = car.state[2]
        ret = [self.raycast(x, y, t) for t in range(angle + LIDAR_MIN, angle + LIDAR_MAX, LIDAR_N-1)]
        ret.append(self.raycast(x, y, angle + LIDAR_MAX))
        return ret

    #returns a list of the other cars sorted by distance
    def nearby_cars(self, car):
        ret = []
        for other in self.cars:
            if other is not car:
                ret.append(other)
        return sorted(ret, lambda c: (c.state[0] - car.state[0])**2 + (c.state[1] - car.state[1])**2)



    def check_collisions(self):
        ret = False
        for i in range(len(self.cars)):
            for j in range(i, len(self.cars)):
                for seg1 in cars[i].get_segments():
                    for seg2 in cars[j].get_segments():
                        if intersect_segments(seg1, seg2):
                            cars[i].collided = True
                            cars[j].collided = True
                            ret = True
        return False

    def step(self):
        for car in self.cars:
            car.step()
        if self.check_collisions():
            #the cars stop moving if collided
            #some naive handling, like set the NN cost to infinity
            pass
        #integrate map collisions here

    def get_cost():
        pass
    def __init__(self):
        pass
