import numpy as np
import scipy.optimize as opt
from shapely import make_valid
from shapely.geometry import Polygon, Point


def point_to_edge_distance(point, edge):
    """Calculate the perpendicular distance from a point to a line segment."""
    p, a, b = np.array(point), np.array(edge[0]), np.array(edge[1])
    
    # Vector from A to B and A to P
    ab = b - a
    ap = p - a
    if np.sum(ab == 0): # means edge is actually a point
        return np.linalg.norm(p - a)

    # Project AP onto AB to find closest point on segment
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    closest_point = a + t * ab
    
    return np.linalg.norm(p - closest_point)


def min_distance_to_edges(point, edges):
    """Returns the minimum distance from a point to all edges in the polygon."""
    return min(point_to_edge_distance(point, edge) for edge in edges)


def incenter_of_polygon(polygon_points, grid_size=100, debug=True): # TODO: new data added, debug should be changed to False
    """Finds the incenter of an irregular polygon by maximizing the minimum distance to edges."""
    polygon = Polygon(polygon_points)
    if not polygon.is_valid:
        if debug:
            polygon = polygon.buffer(0)
        else:
            raise ValueError("Invalid polygon: Ensure the points form a closed, non-intersecting polygon.")
    
    # Get polygon edges
    edges = [(polygon_points[i], polygon_points[(i + 1)%len(polygon_points)]) for i in range(len(polygon_points))]
    
    # Find a rough estimate by sampling grid points inside the polygon
    minx, miny, maxx, maxy = polygon.bounds
    x_vals = np.linspace(minx, maxx, grid_size)
    y_vals = np.linspace(miny, maxy, grid_size)
    
    best_point = None
    best_distance = -np.inf
    
    for x in x_vals:
        for y in y_vals:
            point = (x, y)
            if polygon.contains(Point(point)):
                dist = min_distance_to_edges(point, edges)
                if dist > best_distance:
                    best_distance = dist
                    best_point = point
    
    return best_point

# Example Usage
if __name__ == "__main__":
    polygon_points = [(0, 0), (2, 5), (4, 1), (5, 5), (2, 7), (-1, 5), (-2, 2), (0, 0)]  # Ensure it's closed
    incenter = incenter_of_polygon(polygon_points)
    print("Incenter (Center of Largest Inscribed Circle):", incenter)

    from matplotlib import pyplot as plt
    plt.figure()
    pts = np.array(polygon_points)
    plt.plot(pts[:, 0], pts[:, 1], 'b-')
    plt.plot(incenter[0], incenter[1], 'ro')
    plt.show()



    # def calculate_centroid(points):
    #     x_coords = [point[0] for point in points]
    #     y_coords = [point[1] for point in points]
    #     centroid_x = sum(x_coords) / len(x_coords)
    #     centroid_y = sum(y_coords) / len(y_coords)
    #     return centroid_x, centroid_y
