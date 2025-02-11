import cv2
import json
from numba import jit
import numpy as np
from utils.IncenterCentroid import incenter_of_polygon

@jit(nopython=True)
def line_intersection(l1p1_x, l1p1_y, l1p2_x, l1p2_y, l2p1_x, l2p1_y, l2p2_x, l2p2_y):
    """
    Finds the intersection point of two line segments.

    Parameters:
        (x1, y1, x2, y2): Coordinates of the first line segment.
        (x3, y3, x4, y4): Coordinates of the second line segment.

    Returns:
        Tuple (x, y) if the lines intersect within the segment boundaries, otherwise None.
    """
    # get data
    x1, y1 = l1p1_x, l1p1_y
    x2, y2 = l1p2_x, l1p2_y
    x3, y3 = l2p1_x, l2p1_y
    x4, y4 = l2p2_x, l2p2_y

    # Compute the determinants
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Check if lines are parallel (denominator is zero)
    if denominator == 0:
        return False, (0.0, 0.0)  # No intersection or collinear

    # Compute intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    # Check if intersection is within the segment bounds
    tolerance = 10**-6
    if (min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance and 
        min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance and
        min(x3, x4) - tolerance <= px <= max(x3, x4) + tolerance and 
        min(y3, y4) - tolerance <= py <= max(y3, y4) + tolerance):
        return True, (px, py)

    return False, (0.0, 0.0)  # Intersection is outside segment bounds

@jit(nopython=True)
def find_start_point_jit(centroid, points):
    l2p1_x, l2p1_y = centroid[0], 0
    l2p2_x, l2p2_y = centroid[0], centroid[1]
    for i in range(points.shape[0]):
        found, point = line_intersection(points[i, 0], points[i, 1], 
                                         points[(i+1) % len(points), 0], points[(i+1) % len(points), 1], 
                                         l2p1_x, l2p1_y, 
                                         l2p2_x, l2p2_y)
        if found:
            return True, point
        
    return False, (0.0, 0.0)

def find_start_point(centroid, points, debug=False, img_fp=None):
    l2p1 = centroid[0], 0
    l2p2 = centroid
    for i in range(len(points)):
        l1p1 = points[i]
        l1p2 = points[(i+1) % len(points)]
        found, point = line_intersection(l1p1[0], l1p1[1], l1p2[0], l1p2[1], l2p1[0], l2p1[1], l2p2[0], l2p2[1])

        if found:
            return True, point
        
        if debug:
            # VISUALIZATION
            from matplotlib import pyplot as plt
            print("l1p1", l1p1)
            print("l1p2", l1p2)
            print("l2p1", l2p1)
            print("l2p2", l2p2)
            print("point", point)
            plt.figure()
            plt.imshow(cv2.imread(img_fp)[:, :, ::-1])
            plt.plot([l1p1[0], l1p2[0]], [l1p1[1], l1p2[1]], color='red')
            plt.scatter([centroid[0]], [centroid[1]])
            plt.plot([centroid[0], centroid[0]], [0, centroid[1]])
            plt.show()    
            
    if not debug:
        return False, (0.0, 0.0)

# since all labels with multiple same-class objects has been filtered out
def read_json_as_dict(json_fp, jit=True) -> dict:
    with open(json_fp, 'r') as file:
        data = json.load(file)
    
    for shpidx, shp in enumerate(data['shapes']):
        shp['shpidx'] = shpidx

        # centroid
        try:
            shp['centroid'] = incenter_of_polygon(shp['points'], jit=jit)
        except ValueError as e:
            print("Error:", e)
            print(json_fp)
            shp['centroid'] = incenter_of_polygon(shp['points'], jit=jit, debug=True) # TODO
            assert False
        
        # find_start_point
        if jit:
            found, start_point = find_start_point_jit(np.array(shp['centroid'], np.float64), np.array(shp['points'], np.float64))
        else:
            found, start_point = find_start_point(shp['centroid'], shp['points'])

        if found:
            shp['start_point'] = (float(start_point[0]), float(start_point[1]))
        else:
            print(json_fp)
            assert False, "Start point not found"

    return data
