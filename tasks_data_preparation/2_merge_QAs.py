import os
import json
import glob

if __name__ == "__main__":
    # Path to metadata files
    with open("datafiles.json", "r") as f:
        datafiles = json.load(f)['production']

    for datafile in datafiles:
        meta_fp_dist = datafile['all_qapair_fp']
        json_fps_task = glob.glob(os.path.join(datafile['imgqa_dir'], "*", "*.json"))
        json_fps_task4 = glob.glob(os.path.join(datafile['imgqa_dir'], "*", "*", "*.json"))

        print("json_fps_task", len(json_fps_task))
        print("json_fps_task4", len(json_fps_task4))

    all_qapairs = []
    for json_fp in json_fps_task:
        with open(json_fp, "r") as f:
            all_qapairs.append(json.load(f))

    for json_fp in json_fps_task4:
        with open(json_fp, "r") as f:
            all_qapairs.append(json.load(f))

    with open(meta_fp_dist, "w") as f:
        json.dump(all_qapairs, f)












    # meta_fp_dist = datafiles['all_qapair_fp']
    # meta_fps = [
    #     datafiles['task1_qapair_fp'],
    #     datafiles['task2_qapair_fp']
    # ]

    # # Load metadata from JSON files
    # metadata = []
    # for meta_fp in meta_fps:
    #     with open(meta_fp, "r") as f:
    #         metadata.extend(json.load(f))  # Assuming metadata is a list of {image_path, question, answer}

    # with open(meta_fp_dist, "w") as f:
    #     json.dump(metadata, f)
