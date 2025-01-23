import json

if __name__ == "__main__":
    # Path to metadata files
    meta_fps = [
        "../../coco_dataset/val2017_2_task1_qapair.json",
        "../../coco_dataset/val2017_2_task2_qapair.json"
    ]
    meta_fp_dist = "../../coco_dataset/val2017_3_all_qapairs.json"

    # Load metadata from JSON files
    metadata = []
    for meta_fp in meta_fps:
        with open(meta_fp, "r") as f:
            metadata.extend(json.load(f))  # Assuming metadata is a list of {image_path, question, answer}

    with open(meta_fp_dist, "w") as f:
        json.dump(metadata, f)
