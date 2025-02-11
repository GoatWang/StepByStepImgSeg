import os
import cv2
import json
import math
import numpy as np
from pathlib import Path
from task0_utils import read_json_as_dict
from sklearn.metrics import pairwise_distances

def draw_clock_ticks(img, center, tick_length, n_ticks):
    """
    Draws clock ticks (radial lines) on the given img using OpenCV.

    Parameters:
      img: A NumPy array representing the img.
      center: A tuple (x, y) representing the center of the circle.
      tick_length: The length of each tick line in pixels.
      n_ticks: Number of ticks to draw (default is 12, which divides the circle into 12 equal sectors).
    """
    cx, cy = center
    for i in range(n_ticks):
        angle_rad = math.radians(i * (360 / n_ticks) - 90)
        # Compute the end point for the tick line.
        x_end = int(cx + tick_length * math.cos(angle_rad))
        y_end = int(cy + tick_length * math.sin(angle_rad))
        cv2.line(img, (cx, cy), (x_end, y_end), (0, 0, 255), 2)
        x_end = int(cx + tick_length * 1.5 * math.cos(angle_rad))
        y_end = int(cy + tick_length * 1.5 * math.sin(angle_rad))
        cv2.putText(img, str(i), (x_end, y_end), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def find_closest_tickidx(center, next_pt, gap, n_ticks):
    """
    Find the index of the closest tick line to the given center point.

    Parameters:
      next_pt: A tuple (x, y) representing the center of next point of the circle.
      gap: The length of each tick line in pixels.
      n_ticks: Number of ticks to draw (default is 12, which divides the circle into 12 equal sectors).

    Returns:
      int: The index of the closest tick line.
    """
    cx, cy = center
    min_dist = float("inf")
    closest_tick = 0
    tick_points = []
    for i in range(n_ticks):
        angle_rad = math.radians(i * (360 / n_ticks) - 90)
        x_end = int(cx + gap * math.cos(angle_rad))
        y_end = int(cy + gap * math.sin(angle_rad))
        tick_points.append((x_end, y_end))
    tick_points = np.array(tick_points)
    return np.argmin(pairwise_distances([next_pt], tick_points)[0])

def draw_history_trajectory_and_clock(img_fp, pts, ptidx, gap, tick_length, n_ticks=20):
    img_dst = cv2.imread(img_fp)
    pts = np.array(pts, dtype=int)
    cv2.polylines(img_dst, [pts[:ptidx].reshape(-1, 1, 2)], 
                  isClosed=False, color=(0, 0, 0), thickness=2)
    draw_clock_ticks(img_dst, pts[ptidx-1], tick_length=tick_length, n_ticks=n_ticks)
    tickidx = find_closest_tickidx(pts[ptidx-1], pts[ptidx], gap, n_ticks=n_ticks)
    return img_dst, tickidx

def create_task4_qapair(img_fp, shp, img_fp_dst, pt_id_str, n_ticks, tickidx):
    id_str = os.path.basename(img_fp).replace(".jpg", "") + "_task4_" + str(shp['shpidx']) + "_" + pt_id_str + "_" + shp['label']
    qa_pair = {
        "id": id_str,
        "image": img_fp_dst,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nThe image shows an incomplete annotated polygon (black line) enclosing the {shp['label']}. The polygon should be completed in a clockwise direction. There are 12 red lines indicating possible directions for the next point, numbered sequentially in a clockwise manner starting from 0 at the top and ending at {n_ticks-1}. Based on the pattern, continuity, and shape of {shp['label']}, identify the most probable direction for the next point. Please respond with a number from 0 to {n_ticks-1}."
            },
            {
                "from": "gpt",
                "value": f"{tickidx}"
            },
        ]
    }
    return qa_pair


if __name__ == "__main__":
    import os
    import json
    import glob
    import numpy as np
    from PIL import Image
    
    def get_image_size(file_path):
        with Image.open(file_path) as img:
            return img.size  # (width, height)

    with open("datafiles.json", "r") as f:
        datafiles = json.load(f)['preview']

    for datafile in datafiles:
        img_dir = datafile['imgmask_filtered_dir']
        mask_dir = datafile['task4_1_samedist_gap_dir']
        task4_img_dir = datafile['task4_2_trajectory_poly_dir']
        qapair_fp = datafile['task4_3_qapair_fp']
        Path(task4_img_dir).mkdir(exist_ok=True, parents=True)

        img_fps = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:10]
        get_mask_fp = lambda img_fp: os.path.join(mask_dir, os.path.basename(img_fp).replace(".jpg", ".json"))
        mask_fn = os.path.basename(img_fps[0]).replace(".jpg", ".json")
        img_mask_fp_pairs = [(img_fp, get_mask_fp(img_fp)) for img_fp in img_fps]
        
        qa_pairs = []
        for idx, (img_fp, mask_fp) in enumerate(img_mask_fp_pairs[:]):
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(img_mask_fp_pairs)}")

            save_dir_img = os.path.join(task4_img_dir, os.path.basename(img_fp).replace(".jpg", ""))
            Path(save_dir_img).mkdir(exist_ok=True, parents=True)

            h, w = get_image_size(img_fp)
            mask_data = read_json_as_dict(mask_fp)
            for shp in mask_data['shapes']:
                label = shp['label']
                points = shp['points']
                obj_name = shp['label']

                for ptidx in range(1, len(points)):
                    img_fp_dst = os.path.join(save_dir_img, os.path.basename(img_fp).replace(".jpg", f"_task4_{shp['shpidx']:02d}_{ptidx:04d}_{obj_name}.jpg"))

                    # 1. draw the previous points
                    # 2. draw clock ticks
                    # 3. put text on each tickline
                    n_ticks = 12
                    gap = datafile['gap']/336*np.sqrt(h * w)
                    tick_length = gap * 2
                    img_dst, tickidx = draw_history_trajectory_and_clock(img_fp, points, ptidx, gap, tick_length, n_ticks=n_ticks)
                    cv2.imwrite(img_fp_dst, img_dst)

                    # 4. produce question-answer pair
                    pt_id_str = str(ptidx).zfill(5)
                    qa_pair = create_task4_qapair(img_fp, shp, img_fp_dst, pt_id_str, n_ticks, tickidx)

                    with open(img_fp_dst.replace(".jpg", ".json"), "w") as f:
                        json.dump(qa_pair, f, indent=4)
                    qa_pairs.append(qa_pair)

        with open(qapair_fp, 'w') as f:
            json.dump(qa_pairs, f, indent=4)

    # {
    #     "id": "000000000285_task2_00_bear",
    #     "image": "/home/wanghsuanchung/Projects/StepByStepImgSeg/coco_dataset/val2017_1_imgmask_filtered/000000000285.jpg",
    #     "conversations": [
    #         {
    #             "from": "human",
    #             "value": "<image>\nCan you identify the center of bear is in which block (from 0 to 9). Please reply just one number."
    #         },
    #         {
    #             "from": "gpt",
    #             "value": "7"
    #         }
    #     ]
    # },








