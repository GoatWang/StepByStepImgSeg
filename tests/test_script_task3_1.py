import os
import sys
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.tasks_data_prepare.task3_1_rearrange_points_gap import adjust_points_gap

if __name__ == "__main__":
    result_dir = os.path.join("temp", "test_script_result")
    Path(result_dir).mkdir(exist_ok=True, parents=True)

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




