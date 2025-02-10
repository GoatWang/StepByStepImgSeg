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
from tasks_data_preparation.task4_1_rearrange_points_gap import adjust_points_gap

def get_image_size(file_path):
    with Image.open(file_path) as img:
        return img.size  # (width, height)

if __name__ == "__main__":
    img_mask_dir = os.path.join(root_dir, "tests", "test_data", "val2017_1_imgmask_filtered")
    result_dir = os.path.join(root_dir, "temp", "test_script_result", "task4_1_rearrange_points_gap")
    Path(result_dir).mkdir(exist_ok=True, parents=True)

    img_fps = glob.glob(os.path.join(img_mask_dir, "*.jpg"))
    get_mask_fp = lambda img_fp: img_fp.replace(".jpg", ".json")
    img_mask_fps = [(img_fp, get_mask_fp(img_fp)) for img_fp in img_fps]

    # generate 10 random colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (94, 17, 21), (99, 24, 122), (234, 174, 238),
        (223, 23, 101), (145, 108, 62), (195, 218, 109),
        (12, 171, 183), (17, 27, 5), (212, 184, 147), (20, 113, 34)]  # Example colors: red, green, blue

    for img_fp, mask_fp in img_mask_fps:
        h, w = get_image_size(img_fp)
        mask_fp_dst = os.path.join(result_dir, os.path.basename(mask_fp))
        adjust_points_gap(mask_fp, mask_fp_dst, gap=10/336*np.sqrt(h * w))

        # draw the mask with points order id
        # draw src image
        img = cv2.imread(img_fp)
        with open(mask_fp, "r") as f:
            anno_data = json.load(f)
        for idx, shape in enumerate(anno_data["shapes"]):
            points = shape["points"]
            cv2.drawContours(img, [np.array(points, np.int32)], -1, tuple(colors[idx % len(colors)]), 2)
            for i, (x, y) in enumerate(points):
                cv2.putText(img, str(i), (int(x) + 5, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(os.path.join(result_dir, os.path.basename(img_fp)), img)

        # draw dst image
        img_dst = cv2.imread(img_fp)
        with open(mask_fp_dst, "r") as f:
            data = json.load(f)
        for idx, shape in enumerate(data["shapes"]):
            points = shape["points"]
            cv2.drawContours(img_dst, [np.array(points, np.int32)], -1, tuple(colors[idx % len(colors)]), 2)
            for i, (x, y) in enumerate(points):
                cv2.putText(img_dst, str(i), (int(x) + 5, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(os.path.join(result_dir, os.path.basename(img_fp.replace(".jpg", "_mod.jpg"))), img_dst)




