import os
import cv2
import glob 
import json
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from task0_utils import read_json_as_dict
from sklearn.metrics.pairwise import pairwise_distances

def draw_ruler_task3(img, shp, tick_gap=15):
    x0, y0 = shp['centroid']
    x1, y1 = shp['centroid'][0], 0
    
    ticks = []
    y_tmp = y0
    while y_tmp > y1:
        y_tmp -= tick_gap
        ticks.append((x0, y_tmp))

    cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    for tick_idx, tick in enumerate(ticks):
        cv2.line(img, (int(tick[0]), int(tick[1])), (int(tick[0] + 10), int(tick[1])), (0, 0, 255), 1)
        cv2.putText(img, str(tick_idx), (int(tick[0] + 15), int(tick[1]+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    nearest_tick_idx = np.argmin(pairwise_distances([shp['start_point']], ticks)[0])
    return img, nearest_tick_idx

def create_task3_qapair(img_fp, shp, img_fp_dst, nearest_tick_idx): # shp_idx, obj_name
    id_str = os.path.basename(img_fp).replace(".jpg", "") + "_task3_" + str(shp['shpidx']).zfill(2) + "_" + shp['label']
    qa_pair = {
        "id": id_str,
        "image": img_fp_dst,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nA vertical ruler is drawn from the center of the {shp['label']} to the top, with each tick spaced 15 pixels apart. The ticks are numbered starting from 0 at the bottom, increasing upwards, with the ID noted on the right side of each tick. Identify the tick number that is closest to the boundary of the {shp['label']}. Please provide only the tick ID."
            },
            {
                "from": "gpt",
                "value": f"{nearest_tick_idx}"
            },
        ]
    }
    return qa_pair

# 000000479126_task3_00_laptop
if __name__  == "__main__":
    with open("datafiles.json", "r") as f:
        datafiles = json.load(f)['preview']

    for datafile in datafiles:
        imgmask_dir = datafile['imgmask_filtered_dir']
        task3_img_dir = datafile['task3_1_img_dir']
        task3_qapair_fp = datafile['task3_2_qapair_fp']
        Path(task3_img_dir).mkdir(parents=True, exist_ok=True)

        img_fps = sorted(glob.glob(os.path.join(imgmask_dir, "*.jpg")))[:10]
        mask_fps = [os.path.join(imgmask_dir, os.path.basename(img_fp).replace(".jpg", ".json")) for img_fp in img_fps]

        qa_pairs = []
        for idx, (img_fp, mask_fp) in enumerate(zip(img_fps, mask_fps)):
            if idx % 100 == 0:
                print(idx, "/", len(img_fps))

            img = cv2.imread(img_fp)
            mask_data = read_json_as_dict(mask_fp)

            for shp in mask_data['shapes']:
                img_dst, nearest_tick_idx = draw_ruler_task3(img.copy(), shp, tick_gap=15)
                img_fp_dst = os.path.join(task3_img_dir, os.path.basename(img_fp).replace(".jpg", f"_task3_{shp['shpidx']:02d}_{shp['label']}.jpg"))
                qa_pair = create_task3_qapair(img_fp, shp, img_fp_dst, nearest_tick_idx)
                cv2.imwrite(img_fp_dst, img_dst)

                json_fp_dst = os.path.join(task3_img_dir, os.path.basename(img_fp).replace(".jpg", f"_task3_{shp['shpidx']:02d}_{shp['label']}.json"))
                with open(json_fp_dst, "w") as f:
                    json.dump(qa_pair, f, indent=4)
                qa_pairs.append(qa_pair)
                
        with open(task3_qapair_fp, "w") as f:
            json.dump(qa_pairs, f, indent=4)

# Read metadata (QA pairs)
# Example metadata.json structure:
# [
#   {
#     "id": "997bb945-628d-4724-b370-b84de974a19f",
#     "image": "part-000001/997bb945-628d-4724-b370-b84de974a19f.jpg",
#     "conversations": [
#       {
#         "from": "human",
#         "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image."
#       },
#       {
#         "from": "gpt",
#         "value": "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render. "
#       },
#     ]
#   },
#   ...
# ]
