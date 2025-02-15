import os
import cv2
import sys
import json
import glob
import numpy as np
from pathlib import Path
sys.path.append(os.path.dirname(__file__))
from task0_utils import read_json_as_dict   
from task1_1_horizontal_locate import draw_vertical_lines_and_blocks
from task1_2_QA_pair_generation import sort_imgmask_into_qa_pair_task1
from task2_1_vertical_locate import draw_horizontal_lines_and_blocks
from task2_2_QA_pair_generation import sort_imgmask_into_qa_pair_task2
from task3_1_start_point_locate import draw_ruler_task3, create_task3_qapair
from task4_1_rearrange_points_gap import adjust_points_gap
from task4_2_draw_clock_ticks import draw_history_trajectory_and_clock, create_task4_qapair

def save_img_and_qapair(img, qa_pair, img_fp_dst, qa_pair_fp_dst):
    cv2.imwrite(img_fp_dst, img)
    with open(qa_pair_fp_dst, "w") as f:
        json.dump(qa_pair, f, indent=4)


def main():
    with open(os.path.join(os.path.dirname(__file__), "datafiles.json"), "r") as f:
        datafiles = json.load(f)['production']

    for datafile in datafiles:
        imgmask_dir = datafile['imgmask_filtered_dir']
        imgqa_dir = datafile['imgqa_dir']
        img_fps = sorted(glob.glob(os.path.join(imgmask_dir, "*.jpg")))
        mask_fps = sorted(glob.glob(os.path.join(imgmask_dir, "*.json")))
        for idx, (img_fp, mask_fp) in enumerate(zip(img_fps, mask_fps)):
            if idx % 100 == 0:
                print(idx, "/", len(img_fps))

            img = cv2.imread(img_fp)    
            mask_data = read_json_as_dict(mask_fp)

            for shp in mask_data['shapes']:
                shp_task_dir = os.path.join(imgqa_dir, os.path.basename(img_fp).replace(".jpg", f"_{shp['shpidx']:02d}_{shp['label']}"))
                Path(shp_task_dir).mkdir(exist_ok=True, parents=True)
                get_img_fp_dst = lambda img_fp, task_id, shp: os.path.abspath(os.path.join(shp_task_dir, os.path.basename(shp_task_dir)+ f"_{task_id}.jpg"))
                
                # task1
                img_fp_task1 = get_img_fp_dst(img_fp, "task1", shp)
                img_task1 = draw_vertical_lines_and_blocks(img.copy())
                qa_pair_task1 = sort_imgmask_into_qa_pair_task1(img_fp, shp, img_fp_task1, img.shape)
                save_img_and_qapair(img_task1, qa_pair_task1, img_fp_task1, img_fp_task1.replace(".jpg", ".json"))
                
                # task2
                img_fp_task2 = get_img_fp_dst(img_fp, "task2", shp)
                img_task2 = draw_horizontal_lines_and_blocks(img.copy())
                qa_pair_task2 = sort_imgmask_into_qa_pair_task2(img_fp, shp, img_fp_task2, img.shape)
                save_img_and_qapair(img_task2, qa_pair_task2, img_fp_task2, img_fp_task2.replace(".jpg", ".json"))

                # task3
                img_fp_task3 = get_img_fp_dst(img_fp, "task3", shp)
                img_task3, nearest_tick_idx = draw_ruler_task3(img.copy(), shp, tick_gap=15)
                qa_pair_task3 = create_task3_qapair(img_fp, shp, img_fp_task3, nearest_tick_idx)
                save_img_and_qapair(img_task3, qa_pair_task3, img_fp_task3, img_fp_task3.replace(".jpg", ".json"))

                # task4
                h, w = img.shape[:2]
                gap = datafile['gap']/336*np.sqrt(h * w)
                points_adjusted = adjust_points_gap(shp, gap)
                img_dir_task4 = os.path.join(shp_task_dir, os.path.basename(shp_task_dir)+ f"_task4")
                Path(img_dir_task4).mkdir(exist_ok=True, parents=True)
                for ptidx in range(1, len(points_adjusted)):
                    pt_id_str = str(ptidx).zfill(5)
                    img_fp_task4_pt = os.path.abspath(os.path.join(img_dir_task4, os.path.basename(img_dir_task4) + f"_{pt_id_str}" + ".jpg"))

                    n_ticks = 12
                    tick_length = gap * 2
                    img_task4_pt, tickidx = draw_history_trajectory_and_clock(img_fp, points_adjusted, ptidx, gap, tick_length, n_ticks=n_ticks)
                    qa_pair_task4_pt = create_task4_qapair(img_fp, shp, img_fp_task4_pt, pt_id_str, n_ticks, tickidx)
                    save_img_and_qapair(img_task4_pt, qa_pair_task4_pt, img_fp_task4_pt, img_fp_task4_pt.replace(".jpg", ".json"))



if __name__ == "__main__": 
    main()