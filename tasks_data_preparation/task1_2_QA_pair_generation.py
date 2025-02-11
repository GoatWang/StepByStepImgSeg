import os
import cv2
import glob 
import json
import numpy as np
from matplotlib import pyplot as plt
from task0_utils import read_json_as_dict

def sort_imgmask_into_qa_pair_task1(img_fp, shp, img_fp_dst, img_shape):
    """
    img: used to calculate the width adn height
    img_fp: used to generate the id the the qa pair
    """
    height, width = img_shape[:2]
    height_blk, width_blk = height / 10, width / 10

    id_str = os.path.basename(img_fp).replace(".jpg", "") + "_task1_" + str(shp['shpidx']).zfill(2) + "_" + shp['label']
    qa_pair = {
        "id": id_str,
        "image": img_fp_dst,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nThe image is vertically divided into 10 equal sections, with each section numbered from 0 to 9, from left to right. Based on this division, identify which numbered block contains the center of the {shp['label']}. Please reply with only a single number."
            },
            {
                "from": "gpt",
                "value": f"{int(shp['centroid'][0] // width_blk)}"
            },
        ]
    }
    return qa_pair

if __name__  == "__main__":
    with open("datafiles.json", "r") as f:
        datafiles = json.load(f)['preview']

    for datafile in datafiles:
        imgmask_dir = datafile['imgmask_filtered_dir']
        task1_img_dir = datafile['task1_1_img_dir']
        task1_qapair_fp = datafile['task1_2_qapair_fp']

        qa_pairs = []
        img_fps = sorted(glob.glob(os.path.join(imgmask_dir, "*.jpg")))[:10]
        mask_fps = sorted(glob.glob(os.path.join(imgmask_dir, "*.json")))
        task1_img_fps = sorted(glob.glob(os.path.join(task1_img_dir, "*.jpg")))
        for idx, (img_fp, task1_img_fp, mask_fp) in enumerate(zip(img_fps, task1_img_fps, mask_fps)):
            if idx % 100 == 0:
                print(idx, "/", len(img_fps))

            img_fp = os.path.abspath(img_fp)
            img = cv2.imread(task1_img_fp)
            mask_data = read_json_as_dict(mask_fp)
            for shp in mask_data['shapes']:
                qa_pair = sort_imgmask_into_qa_pair_task1(img_fp, shp, task1_img_fp, img.shape)
                with open(task1_img_fp.replace(".jpg", f"_task1_{shp['shpidx']:02d}_{shp['label']}.json"), "w") as f:
                    json.dump(qa_pair, f, indent=4)
            qa_pairs.append(qa_pair)

            # # # VISUALIZATION
            # height, width = img.shape[:2]
            # height_blk, width_blk = height / 10, width / 10
            # print("img_fp: ", img_fp)
            # print("obj_name: ", shp['label'])
            # print("obj_centroid: ", shp['centroid'])
            # print("points", shp['points'])
            # # print("obj_centroid", obj_centroid)
            # # print("height_blk, width_blk", height_blk, width_blk)
            # # print(obj_centroid[0] // width_blk)
            # # plt.imshow(img)
            # # plt.scatter([obj_centroid[0]], [obj_centroid[1]])
            # # plt.show()

    with open(task1_qapair_fp, 'w') as f:
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
