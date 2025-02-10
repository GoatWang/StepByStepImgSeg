import os
import glob
import shutil
import numpy as np
from pathlib import Path

dirs_src = [
    os.path.join(os.path.dirname(__file__), "..", "coco_dataset", "val2017_1_imgmask_filtered"), 
    os.path.join(os.path.dirname(__file__), "..", "coco_dataset", "val2017_2_task4_1_samedist_gap_dir"), 
]
dirs_dst = [os.path.join(os.path.dirname(__file__), "test_data", os.path.basename(d)) for d in dirs_src]
for dir_dst in dirs_dst:
    Path(dir_dst).mkdir(exist_ok=True, parents=True)

for dir_src, dir_dst in zip(dirs_src, dirs_dst):
    img_fps = sorted(glob.glob(os.path.join(dir_src, "*.jpg")))
    mask_fps = sorted(glob.glob(os.path.join(dir_src, "*.json")))
    data_length = max(len(img_fps), len(mask_fps))

    np.random.seed(2025)
    idxs = np.random.randint(0, data_length, 30)
    for idx in idxs:
        if len(img_fps) > 0:
            img_fp = img_fps[idx]
            shutil.copyfile(img_fp, os.path.join(dir_dst, os.path.basename(img_fp)))
        if len(mask_fps) > 0:
            mask_fp = mask_fps[idx]
            shutil.copyfile(mask_fp, os.path.join(dir_dst, os.path.basename(mask_fp)))



 