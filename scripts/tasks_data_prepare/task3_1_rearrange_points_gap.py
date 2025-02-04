import json
import math

def adjust_points_gap(input_json_fp, output_json_fp, gap):
    """
    Load a LabelMe annotation JSON file, re-sample each polygon (mask) so that
    the points are spaced roughly 'gap' pixels apart along its boundary.
    The final segment is allowed to be shorter than gap.
    
    Parameters:
        input_json_fp (str): Filepath of the input LabelMe JSON annotation.
        output_json_fp (str): Filepath where the modified JSON will be saved.
        gap (float): The desired distance (in pixels) between consecutive points.
    """
    # Load the JSON annotation data.
    with open(input_json_fp, 'r') as f:
        data = json.load(f)
    
    # Process each shape in the annotation.
    for shape in data.get("shapes", []):
        # Skip shapes that do not have points.
        if "points" not in shape:
            continue
        
        pts = shape["points"]
        if len(pts) < 2:
            continue  # Nothing to do for shapes with fewer than 2 points.
        
        # For masks (closed polygons), we ensure the polygon is closed.
        # If there are 3 or more points and the first and last are not the same,
        # we add the first point at the end.
        if len(pts) >= 3:
            if math.dist(pts[0], pts[-1]) > 1e-6:
                pts_closed = pts + [pts[0]]
            else:
                pts_closed = pts
        else:
            pts_closed = pts  # For a line with 2 points, treat as open.
        
        # Compute the lengths of each segment along the boundary.
        seg_lengths = []
        for i in range(1, len(pts_closed)):
            d = math.dist(pts_closed[i-1], pts_closed[i])
            seg_lengths.append(d)
        total_length = sum(seg_lengths)
        
        # Generate positions along the boundary at intervals of 'gap'.
        # We take positions 0, gap, 2*gap, ... up to (but not necessarily equal to) total_length.
        n_full = int(total_length // gap)
        positions = [i * gap for i in range(n_full)]
        # Always add the final position (which may be less than gap away from the previous one)
        if not positions or (total_length - positions[-1]) > 1e-6:
            positions.append(total_length)
        
        # Interpolate new points along the boundary.
        new_points = []
        seg_index = 0          # current segment index
        seg_start_distance = 0 # cumulative distance at the start of current segment
        
        for pos in positions:
            # Advance to the segment that contains the position 'pos'
            while seg_index < len(seg_lengths) and pos > seg_start_distance + seg_lengths[seg_index]:
                seg_start_distance += seg_lengths[seg_index]
                seg_index += 1
            
            # If we somehow run past the last segment, just take the last point.
            if seg_index >= len(seg_lengths):
                new_points.append(pts_closed[-1])
            else:
                # Compute how far along the current segment the new point lies.
                # (t=0 corresponds to the segment start, t=1 to the segment end)
                seg_length = seg_lengths[seg_index]
                # Avoid division by zero.
                t = 0 if seg_length == 0 else (pos - seg_start_distance) / seg_length
                
                # Linearly interpolate between the two segment endpoints.
                p_start = pts_closed[seg_index]
                p_end = pts_closed[seg_index + 1]
                new_x = p_start[0] + t * (p_end[0] - p_start[0])
                new_y = p_start[1] + t * (p_end[1] - p_start[1])
                new_points.append([new_x, new_y])
        
        # For a closed polygon, if the final point is (nearly) identical to the first,
        # remove it so that the polygon remains defined by unique points.
        if len(pts) >= 3 and len(new_points) > 1 and math.dist(new_points[0], new_points[-1]) < 1e-6:
            new_points.pop()
        
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

