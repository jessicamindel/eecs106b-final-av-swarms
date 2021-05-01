import numpy as np


# https://gist.github.com/danieljfarrell/faf7c4cafd683db13cbc
# public domain

def magnitude(vector):
   return np.sqrt(np.dot(np.array(vector),np.array(vector)))

def norm(vector):
   return np.array(vector)/magnitude(np.array(vector))

def intersect_ray_segment(rayOrigin, angle, point1, point2):
    """
    >>> # Line segment
    >>> z1 = (0,0)
    >>> z2 = (10, 10)
    >>>
    >>> # Test ray 1 -- intersecting ray
    >>> r = (0, 5)
    >>> d = norm((1,0))
    >>> len(lineRayIntersectionPoint(r,d,z1,z2)) == 1
    True
    >>> # Test ray 2 -- intersecting ray
    >>> r = (5, 0)
    >>> d = norm((0,1))
    >>> len(lineRayIntersectionPoint(r,d,z1,z2)) == 1
    True
    >>> # Test ray 3 -- intersecting perpendicular ray
    >>> r0 = (0,10)
    >>> r1 = (10,0)
    >>> d = norm(np.array(r1)-np.array(r0))
    >>> len(lineRayIntersectionPoint(r0,d,z1,z2)) == 1
    True
    >>> # Test ray 4 -- intersecting perpendicular ray
    >>> r0 = (0, 10)
    >>> r1 = (10, 0)
    >>> d = norm(np.array(r0)-np.array(r1))
    >>> len(lineRayIntersectionPoint(r1,d,z1,z2)) == 1
    True
    >>> # Test ray 5 -- non intersecting anti-parallel ray
    >>> r = (-2, 0)
    >>> d = norm(np.array(z1)-np.array(z2))
    >>> len(lineRayIntersectionPoint(r,d,z1,z2)) == 0
    True
    >>> # Test ray 6 --intersecting perpendicular ray
    >>> r = (-2, 0)
    >>> d = norm(np.array(z1)-np.array(z2))
    >>> len(lineRayIntersectionPoint(r,d,z1,z2)) == 0
    True
    """
    # Convert to numpy arrays
    rayOrigin = np.array(rayOrigin, dtype=np.float)
    rayDirection = np.array([np.cos(angle), np.sin(angle)])
    point1 = np.array(point1, dtype=np.float)
    point2 = np.array(point2, dtype=np.float)
    
    # Ray-Line Segment Intersection Test in 2D
    # http://bit.ly/1CoxdrG
    v1 = rayOrigin - point1
    v2 = point2 - point1
    v3 = np.array([-rayDirection[1], rayDirection[0]])
    t1 = np.cross(v2, v1) / np.dot(v2, v3)
    t2 = np.dot(v1, v3) / np.dot(v2, v3)
    if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
        return t1
    return -1

# https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

# Return true if line segments AB and CD intersect
def intersect_segments(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


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


    get_segments(self):
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
    
    intersect(self, other):
        for seg1 in self.get_segments():
            for seg2 in other.get_segments():
                if(intersect_segments(seg1, seg2)):
                    return True
        return False

    #apply control inputs, changing velocity
    control(self, v, dphi):
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
    step(self):
        if not self.collided:
            self.state += TIMESTEP * self.velocity

class Sim:
    cars = []

    spawn_car(self, x, y, theta):
        car = Car()
        car.state[0] = x
        car.state[1] = y
        car.state[2] = theta
        self.cars.push(car)

    raycast(self, x, y, angle):
        best = float('inf')
        for car in self.cars:
            for segment in car.get_segments():
                d = intersect_ray_segment([x,y], angle, segment[0], segment[1])
                if d != -1 and d < best:
                    best = d
        return best

    lidar(self, car):
        x = car.state[0]
        y = car.state[1]
        angle = car.state[2]
        ret = [self.raycast(x, y, t) for t in range(angle + LIDAR_MIN, angle + LIDAR_MAX, LIDAR_N-1)]
        ret.append(self.raycast(x, y, angle + LIDAR_MAX))
        return ret

    check_collisions(self):
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

    step(self):
        for car in self.cars:
            car.step()
        if self.check_collisions():
            #the cars stop moving if collided
            #some naive handling, like set the NN cost to infinity
            pass

    get_cost():
        pass
    __init__(self):
        pass

    
