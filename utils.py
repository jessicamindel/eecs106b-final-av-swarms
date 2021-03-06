import numpy as np

def magnitude(vector):
   return np.sqrt(np.dot(np.array(vector),np.array(vector)))

def norm(vector):
   return np.array(vector)/magnitude(np.array(vector))

def rot_matrix(angle):
	'''Returns a 2D counterclockwise rotation matrix.'''
	return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

def lerp(t, x0, x1):
    return (1.0 - t) * x0 + t * x1

def normalize_between(val, min_val, max_val):
    return (max_val - val) / (max_val - min_val)

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
        return t1, rayOrigin + rayDirection * t1
    return -1, None

# https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect_segments(seg1, seg2):
    A = seg1[0]
    B = seg1[1]
    C = seg2[0]
    D = seg2[1]
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def point_in_rect(point, rect_center, rect_angle, rect_width, rect_height, padding=0):
    # Rotate the image and the point
    R = rot_matrix(rect_angle)
    point_R = R @ np.array(point)
    center_R = R @ np.array(rect_center)
    # Check within the bounds of the rectangle
    in_horizontal = np.abs(point_R[0] - center_R[0]) <= rect_width / 2 + padding
    in_vertical = np.abs(point_R[1] - center_R[1]) <= rect_height / 2 + padding
    return in_horizontal and in_vertical
