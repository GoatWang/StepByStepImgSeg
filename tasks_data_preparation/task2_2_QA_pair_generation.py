import os
import cv2
import glob 
import json
import numpy as np
from matplotlib import pyplot as plt
from task0_utils import read_json_as_dict

def sort_imgmask_into_qa_pair_task2(img_fp, shp, task2_img_fp, img_shape):
    """
    img: used to calculate the width adn height
    img_fp: used to generate the id the the qa pair
    """
    height, width = img_shape[:2]
    height_blk, width_blk = height / 10, width / 10
    
    id_str = os.path.basename(img_fp).replace(".jpg", "") + "_task2_" + str(shp['shpidx']).zfill(2) + "_" + shp['label']
    qa_pair = {
        "id": id_str,
        "image": task2_img_fp,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nThe image is divided into 10 equal horizontal sections, numbered from 0 to 9 from top to bottom. Based on this division, identify which numbered block contains the center of the {shp['label']}. Please reply with only a single number."
            },
            {
                "from": "gpt",
                "value": f"{int(shp['centroid'][1] // height_blk)}"
            },
        ]
    }
    return qa_pair

if __name__  == "__main__":
    with open("datafiles.json", "r") as f:
        datafiles = json.load(f)['preview']

    for datafile in datafiles:
        imgmask_dir = datafile['imgmask_filtered_dir']
        task2_img_dir = datafile['task2_1_img_dir']
        task2_qapair_fp = datafile['task2_2_qapair_fp']

        qa_pairs = []
        img_fps = sorted(glob.glob(os.path.join(imgmask_dir, "*.jpg")))[:10]
        mask_fps = sorted(glob.glob(os.path.join(imgmask_dir, "*.json")))
        task2_img_fps = sorted(glob.glob(os.path.join(task2_img_dir, "*.jpg")))
        for idx, (img_fp, task2_img_fp, mask_fp) in enumerate(zip(img_fps, task2_img_fps, mask_fps)):
            if idx % 100 == 0:
                print(idx, "/", len(img_fps))

            img_fp = os.path.abspath(img_fp)
            img = cv2.imread(task2_img_fp)
            mask_data = read_json_as_dict(mask_fp)
            for shp in mask_data['shapes']:
                qa_pair = sort_imgmask_into_qa_pair_task2(img_fp, shp, task2_img_fp, img.shape)

                with open(task2_img_fp.replace(".jpg", f"_task2_{shp['shpidx']:02d}_{shp['label']}.json"), "w") as f:
                    json.dump(qa_pair, f, indent=4)
                qa_pairs.append(qa_pair)

                # print("obj_centroid", obj_centroid)
                # print("height_blk, width_blk", height_blk, width_blk)
                # print(obj_centroid[1] // height_blk)
                # plt.imshow(img)
                # plt.scatter([obj_centroid[0]], [obj_centroid[1]])
                # plt.show()

            with open(task2_qapair_fp, 'w') as f:
                json.dump(qa_pairs, f, indent=4)



# Read metadata (QA pairs)
# Example metadata.json structure:
# {
#     "image1.jpg": [
#         {"question": "What is in the image?", "answer": "A cat"},
#         {"question": "What color is the object?", "answer": "Brown"}
#     ],
#     "image2.jpg": [
#         {"question": "What color is the sky?", "answer": "Blue"}
#     ]
# }
