import os
import cv2
import glob 
import json

# since all labels with multiple same-class objects has been filtered out
def read_json_as_centroid_dict(json_fp) -> dict:
    def calculate_centroid(points):
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)
        return centroid_x, centroid_y
    
    with open(json_fp, 'r') as file:
        data = json.load(file)
    
    centroid_dict = {}
    for shape in data.get('shapes', []):
        label = shape.get('label')
        points = shape.get('points', [])
        if points:
            centroid = calculate_centroid(points)
            centroid_dict[label] = centroid
    
    return centroid_dict

import numpy as np
from matplotlib import pyplot as plt
def sort_imgmask_into_qa_pair_task1(imgmask_dir, task1_img_dir, qa_pairs_file):
    qa_pairs = []
    img_fps = sorted(glob.glob(os.path.join(imgmask_dir, "*.jpg")))
    json_fps = sorted(glob.glob(os.path.join(imgmask_dir, "*.json")))
    task1_img_fps = sorted(glob.glob(os.path.join(task1_img_dir, "*.jpg")))
    for img_fp, task1_img_fp, json_fp in zip(img_fps, task1_img_fps, json_fps):
        img_fp = os.path.abspath(img_fp)
        img = cv2.imread(task1_img_fp)
        height, width = img.shape[:2]
        height_blk, width_blk = height / 10, width / 10

        centroid_dict = read_json_as_centroid_dict(json_fp)
        for obj_name, obj_centroid in centroid_dict.items():
            qa_pair = {
                "image_path": img_fp,
                "question": f"Can you identify the center of {obj_name} is in which block (from 0 to 9). Please reply just one number.",
                "answer": f"{obj_centroid[0] // width_blk}"
            }
            qa_pairs.append(qa_pair)

            # print("obj_centroid", obj_centroid)
            # print("height_blk, width_blk", height_blk, width_blk)
            # print(obj_centroid[0] // width_blk)
            # plt.imshow(img)
            # plt.scatter([obj_centroid[0]], [obj_centroid[1]])
            # plt.show()

    with open(qa_pairs_file, 'w') as f:
        json.dump(qa_pairs, f)
        
if __name__  == "__main__":
    imgmask_dir = "../coco_dataset/val2017_1_imgmask_filtered"
    task1_img_dir = "../coco_dataset/val2017_2_task1_horizontal_locate"
    qa_pairs_file = "../coco_dataset/val2017_2_task1_qapair.json"
    sort_imgmask_into_qa_pair_task1(imgmask_dir, task1_img_dir, qa_pairs_file)



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