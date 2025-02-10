import os
import cv2
import sys
import json
import glob
import numpy as np
from PIL import Image
from pathlib import Path
root_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_dir)
from tasks_data_preparation.task4_2_draw_clock_ticks import draw_history_trajectory_and_clock

def get_image_size(file_path):
    with Image.open(file_path) as img:
        return img.size  # (width, height)

if __name__ == "__main__":


    img_dir = os.path.join(root_dir, "tests", "test_data", "val2017_1_imgmask_filtered")
    mask_dir = os.path.join(root_dir, "tests", "test_data", "val2017_2_task4_1_samedist_gap_dir")
    result_dir = os.path.join(root_dir, "temp", "test_script_result", "task4_2_draw_clock_ticks")
    Path(result_dir).mkdir(exist_ok=True, parents=True)

    img_fps = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_fps = sorted(glob.glob(os.path.join(mask_dir, "*.json")))

    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
    #     (94, 17, 21), (99, 24, 122), (234, 174, 238),
    #     (223, 23, 101), (145, 108, 62), (195, 218, 109),
    #     (12, 171, 183), (17, 27, 5), (212, 184, 147), (20, 113, 34)]  # Example colors: red, green, blue

    for img_fp, mask_fp in zip(img_fps, mask_fps):
        # mask_fp_dst = os.path.join(result_dir, os.path.basename(mask_fp))
        # adjust_points_gap(mask_fp, mask_fp_dst, gap=10/336*np.sqrt(h * w))

        # data reading
        h, w = get_image_size(img_fp)
        img = cv2.imread(img_fp)
        with open(mask_fp, "r") as f:
            data = json.load(f)

        # draw dst image
        ptidx = 10 # test the nth point in the shape (polygon)
        points = data["shapes"][0]['points'] # only test on first shape (polygon)

        # draw the trajectory of previous points
        gap = 10/336*np.sqrt(h * w)
        tick_length = gap * 2
        img_dst, tickidx = draw_history_trajectory_and_clock(img_fp, points, ptidx, gap, tick_length, n_ticks=10)

        # draw the next point
        cv2.circle(img_dst, (int(points[ptidx][0]), int(points[ptidx][1])), 3, (255, 0, 0), -1)

        # draw the answer
        cv2.putText(img_dst, str(tickidx), (int(points[ptidx][0]) + 10, int(points[ptidx][1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        pt_id_str = str(ptidx).zfill(4)
        img_fp_dst = os.path.join(result_dir, os.path.basename(img_fp).replace(".jpg", "_" + pt_id_str + "_mod.jpg")) 
        cv2.imwrite(img_fp_dst, img_dst)




