import json
import math

def force_clockwise(points):
    """
    Ensure that a polygon defined by a list of points is in clockwise order.
    
    Args:
        points (list): A list of [x, y] coordinates defining a polygon.
        
    Returns:
        list: The list of points, reordered (if necessary) to ensure clockwise orientation.
        
    Note:
        In a standard Cartesian coordinate system (y increasing upward), a positive area 
        computed with the shoelace formula indicates a counterclockwise ordering.
    """
    area = 0.0
    n = len(points)
    for i in range(n):
        x_i, y_i = points[i]
        x_next, y_next = points[(i + 1) % n]
        area += (x_i * y_next - x_next * y_i)
    # For standard coordinates, area > 0 means counterclockwise.
    if area > 0:
        points = list(reversed(points))
    return points

def find_start_point(points, image_top=0):
    """
    Find the "start point" of a polygon. The start point is defined as the vertex 
    that is closest to the vertical segment connecting the top of the image (y=image_top)
    and the polygon's centroid.
    
    Args:
        points (list): A list of [x, y] coordinates defining a polygon.
        image_top (float): The y-coordinate corresponding to the top of the image 
                           (default is 0).
                           
    Returns:
        list: The [x, y] coordinates of the vertex chosen as the start point.
    """
    # Compute the centroid using the arithmetic mean.
    cx = sum(pt[0] for pt in points) / len(points)
    cy = sum(pt[1] for pt in points) / len(points)
    centroid = (cx, cy)
    top_point = (cx, image_top)  # Vertical line will have constant x = cx.

    # Helper: distance from point P to segment AB.
    def point_to_segment_distance(P, A, B):
        Ax, Ay = A
        Bx, By = B
        Px, Py = P
        ABx, ABy = Bx - Ax, By - Ay
        APx, APy = Px - Ax, Py - Ay
        ab2 = ABx**2 + ABy**2
        if ab2 == 0:
            return math.dist(P, A)
        t = (APx * ABx + APy * ABy) / ab2
        # Clamp t to the segment [0, 1]
        t = max(0, min(1, t))
        projx = Ax + t * ABx
        projy = Ay + t * ABy
        return math.dist(P, (projx, projy))
    
    # For each vertex, compute the distance to the vertical segment from top_point to centroid.
    min_dist = float('inf')
    start_point = None
    for pt in points:
        d = point_to_segment_distance(pt, top_point, centroid)
        if d < min_dist:
            min_dist = d
            start_point = pt
    return start_point

def adjust_points_gap(input_json_fp, output_json_fp, gap, image_top=0):
    """
    Load a LabelMe annotation JSON file, re-sample each polygon (mask) so that the 
    points are spaced roughly 'gap' pixels apart along its boundary, force the polygon 
    to be in clockwise order, and rotate it so that the start point is first.
    
    The start point is defined as the vertex closest to the vertical segment joining 
    the top of the image (y=image_top) and the polygon's centroid.
    
    Parameters:
        input_json_fp (str): Filepath of the input LabelMe JSON annotation.
        output_json_fp (str): Filepath where the modified JSON will be saved.
        gap (float): The desired distance (in pixels) between consecutive points.
        image_top (float): The y-coordinate representing the top of the image (default 0).
    """
    # Load the JSON annotation data.
    with open(input_json_fp, 'r') as f:
        data = json.load(f)
    
    # Process each shape in the annotation.
    for shape in data.get("shapes", []):
        # Skip shapes that do not have a "points" key.
        if "points" not in shape:
            continue

        pts = shape["points"]
        if len(pts) < 2:
            continue  # Not enough points to work with.

        # Determine if this shape is a closed polygon (3+ points).
        is_closed = False
        if len(pts) >= 3:
            is_closed = True
            # Ensure the polygon is closed.
            if math.dist(pts[0], pts[-1]) > 1e-6:
                pts_closed = pts + [pts[0]]
            else:
                pts_closed = pts.copy()
        else:
            pts_closed = pts  # For a line (2 points), leave as is.

        # Compute segment lengths along the polygon boundary.
        seg_lengths = []
        for i in range(1, len(pts_closed)):
            d = math.dist(pts_closed[i - 1], pts_closed[i])
            seg_lengths.append(d)
        total_length = sum(seg_lengths)

        # Create target positions along the boundary: 0, gap, 2*gap, ...,
        # and always include the final point.
        n_full = int(total_length // gap)
        positions = [i * gap for i in range(n_full)]
        if not positions or (total_length - positions[-1]) > 1e-6:
            positions.append(total_length)

        # Interpolate new points along the boundary.
        new_points = []
        seg_index = 0          # Current segment index.
        seg_start_distance = 0 # Cumulative distance at the start of the current segment.
        for pos in positions:
            # Advance to the segment that contains the current position.
            while seg_index < len(seg_lengths) and pos > seg_start_distance + seg_lengths[seg_index]:
                seg_start_distance += seg_lengths[seg_index]
                seg_index += 1

            # If we ran past the last segment, use the last point.
            if seg_index >= len(seg_lengths):
                new_points.append(pts_closed[-1])
            else:
                seg_length = seg_lengths[seg_index]
                # Avoid division by zero.
                t = 0 if seg_length == 0 else (pos - seg_start_distance) / seg_length
                p_start = pts_closed[seg_index]
                p_end = pts_closed[seg_index + 1]
                new_x = p_start[0] + t * (p_end[0] - p_start[0])
                new_y = p_start[1] + t * (p_end[1] - p_start[1])
                new_points.append([new_x, new_y])

        # For closed polygons, remove duplicate final point (if nearly identical to first).
        if is_closed and len(new_points) > 1 and math.dist(new_points[0], new_points[-1]) < 1e-6:
            new_points.pop()

        # Force clockwise orientation for closed polygons.
        if is_closed and len(new_points) >= 3:
            new_points = force_clockwise(new_points)
            # Determine the start point (vertex closest to the vertical segment from top to centroid).
            start_pt = find_start_point(new_points, image_top=image_top)
            try:
                idx = new_points.index(start_pt)
            except ValueError:
                idx = 0  # Fallback if exact match is not found.
            # Rotate the list so that the start point becomes the first vertex.
            new_points = new_points[idx:] + new_points[:idx]

        # Update the shape's points.
        shape["points"] = new_points

    # Write the updated annotation to the output file.
    with open(output_json_fp, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage:
# adjust_points_gap('input_annotation.json', 'output_annotation.json', 10)
if __name__ == "__main__":
    import os
    import glob
    from pathlib import Path

    with open("datafiles.json", "r") as f:
        datafiles = json.load(f)

    gap = 50
    for datafile in datafiles:
        imgmask_filtered_dir = datafile['imgmask_filtered_dir']
        task3_samedist_gap_dir = datafile['task3_samedist_gap_dir']
        Path(task3_samedist_gap_dir).mkdir(exist_ok=True, parents=True)

        mask_fps_src = sorted(glob.glob(os.path.join(imgmask_filtered_dir, "*.json")))
        for mask_fp_src in mask_fps_src[:]:
            mask_fp_dst = os.path.join(task3_samedist_gap_dir, os.path.basename(mask_fp_src))
            adjust_points_gap(mask_fp_src, mask_fp_dst, gap=gap)

