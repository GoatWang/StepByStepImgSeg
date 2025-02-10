import cv2
import json
from utils.IncenterCentroid import incenter_of_polygon

def line_intersection(l1p1, l1p2, l2p1, l2p2):
    """
    Finds the intersection point of two line segments.

    Parameters:
        (x1, y1, x2, y2): Coordinates of the first line segment.
        (x3, y3, x4, y4): Coordinates of the second line segment.

    Returns:
        Tuple (x, y) if the lines intersect within the segment boundaries, otherwise None.
    """
    # get data
    x1, y1 = l1p1
    x2, y2 = l1p2
    x3, y3 = l2p1
    x4, y4 = l2p2

    # Compute the determinants
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Check if lines are parallel (denominator is zero)
    if denominator == 0:
        return None  # No intersection or they are collinear

    # Compute intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    # Check if intersection is within the segment bounds
    tolerance = 10**-6
    if (min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance and 
        min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance and
        min(x3, x4) - tolerance <= px <= max(x3, x4) + tolerance and 
        min(y3, y4) - tolerance <= py <= max(y3, y4) + tolerance):
        return (px, py)
    
    return None  # Intersection is outside the segment bounds

def find_start_point(centroid, points, debug=False, img_fp=None):
    l2p1 = centroid[0], 0
    l2p2 = centroid
    for i in range(len(points)):
        l1p1 = points[i]
        l1p2 = points[(i+1) % len(points)]
        result = line_intersection(l1p1, l1p2, l2p1, l2p2)

        if result:
            return result
        
        if debug:
            # VISUALIZATION
            from matplotlib import pyplot as plt
            print("l1p1", l1p1)
            print("l1p2", l1p2)
            print("l2p1", l2p1)
            print("l2p2", l2p2)
            print("result", result)
            plt.figure()
            plt.imshow(cv2.imread(img_fp)[:, :, ::-1])
            plt.plot([l1p1[0], l1p2[0]], [l1p1[1], l1p2[1]], color='red')
            plt.scatter([centroid[0]], [centroid[1]])
            plt.plot([centroid[0], centroid[0]], [0, centroid[1]])
            plt.show()    
            
    if not debug:
        assert False, "No intersection found"

# since all labels with multiple same-class objects has been filtered out
def read_json_as_dict(json_fp) -> dict:
    with open(json_fp, 'r') as file:
        data = json.load(file)
    
    for shpidx, shp in enumerate(data['shapes']):
        try:
            shp['centroid'] = incenter_of_polygon(shp['points'])
        except ValueError as e:
            print("Error:", e)
            print(json_fp)
            shp['centroid'] = incenter_of_polygon(shp['points'], debug=True) # TODO
        shp['start_point'] = find_start_point(shp['centroid'], shp['points'])
        shp['shpidx'] = shpidx
            
    return data
