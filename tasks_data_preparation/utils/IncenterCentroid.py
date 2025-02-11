import numpy as np
from numba import jit
import scipy.optimize as opt
from shapely.geometry import Polygon, Point

@jit(nopython=True)
def point_to_edge_distance_jit(point, edge0, edge1):
    """Calculate the perpendicular distance from a point to a line segment."""
    p = point
    a = edge0
    b = edge1
    
    # Vector from A to B and A to P
    ab = b - a
    ap = p - a
    if np.sum(ab == 0):  # means edge is actually a point
        return np.sqrt(np.sum((p - a) ** 2))

    # Project AP onto AB to find closest point on segment
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = max(0, min(1, t))  # equivalent to np.clip in numba
    closest_point = a + t * ab
    
    return np.sqrt(np.sum((p - closest_point) ** 2))

@jit(nopython=True)
def min_distance_to_edges_jit(point, edges):
    """Returns the minimum distance from a point to all edges in the polygon."""
    min_dist = np.float64(np.inf)
    for i in range(edges.shape[0]):
        dist_new = np.float64(point_to_edge_distance_jit(point, edges[i, 0], edges[i, 1]))
        if dist_new < min_dist:
            min_dist = dist_new

    return min_dist

def point_to_edge_distance(point, edge0, edge1):
    """Calculate the perpendicular distance from a point to a line segment."""
    p, a, b = np.array(point), np.array(edge0), np.array(edge1)
    
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
    return min(point_to_edge_distance(point, edge[0], edge[1]) for edge in edges)

def incenter_of_polygon(polygon_points, grid_size=100, jit=True, debug=True): # TODO: new data added, debug should be changed to False
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
                if jit:
                    point = np.array(point, dtype=np.float64)
                    edges = np.array(edges, dtype=np.float64)
                    dist = min_distance_to_edges_jit(point, edges)
                else:
                    dist = min_distance_to_edges(point, edges)
                if dist > best_distance:
                    best_distance = dist
                    best_point = point
    
    return np.array(best_point).astype(float).tolist()

if __name__ == "__main__":
    import time
    from matplotlib import pyplot as plt
    # Function to generate a random polygon with noise
    def generate_random_polygon(num_points=50, noise_level=0.05):
        """Generates a polygon using a random radius with added sinusoidal noise."""
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        radius = np.random.uniform(0.8, 1.2, size=num_points)  # Randomize radius
        noise = np.sin(angles * np.random.randint(1, 5)) * noise_level  # Add sine noise
        noisy_radius = radius + noise
        points = np.column_stack((np.cos(angles) * noisy_radius, np.sin(angles) * noisy_radius))
        return points    
    
    num_tests = 5
    polygons = [generate_random_polygon() for _ in range(num_tests)]

    # VISUALIZE
    # incenter = incenter_of_polygon(polygons[0])
    # print("Incenter (Center of Largest Inscribed Circle):", incenter)
    # from matplotlib import pyplot as plt
    # plt.figure()
    # pts = np.array(polygons[0])
    # plt.plot(pts[:, 0], pts[:, 1], 'b-')
    # plt.plot(incenter[0], incenter[1], 'ro')
    # plt.show()

    # Measure time for JIT-enabled version
    start_jit = time.time()
    jit_results = [incenter_of_polygon(poly, jit=True) for poly in polygons]
    end_jit = time.time()
    jit_time = end_jit - start_jit

    # Measure time for non-JIT version
    start_no_jit = time.time()
    no_jit_results = [incenter_of_polygon(poly, jit=False) for poly in polygons]
    end_no_jit = time.time()
    no_jit_time = end_no_jit - start_no_jit

    # Print results
    print(f"JIT-enabled execution time: {jit_time:.4f} seconds")
    print(f"Non-JIT execution time: {no_jit_time:.4f} seconds")
    print(f"Speedup: {no_jit_time / jit_time:.2f}x")


# # Example Usage
# if __name__ == "__main__":
#     polygon_points = [(0, 0), (2, 5), (4, 1), (5, 5), (2, 7), (-1, 5), (-2, 2), (0, 0)]  # Ensure it's closed
#     incenter = incenter_of_polygon(polygon_points)
#     print("Incenter (Center of Largest Inscribed Circle):", incenter)

#     from matplotlib import pyplot as plt
#     plt.figure()
#     pts = np.array(polygon_points)
#     plt.plot(pts[:, 0], pts[:, 1], 'b-')
#     plt.plot(incenter[0], incenter[1], 'ro')
#     plt.show()



#     # def calculate_centroid(points):
#     #     x_coords = [point[0] for point in points]
#     #     y_coords = [point[1] for point in points]
#     #     centroid_x = sum(x_coords) / len(x_coords)
#     #     centroid_y = sum(y_coords) / len(y_coords)
#     #     return centroid_x, centroid_y
