import json

if __name__ == "__main__":
    # Path to metadata files
    with open("datafiles.json", "r") as f:
        datafiles = f.read()['production']

    meta_fps = [
        datafiles['task1_qapair_fp'],
        datafiles['task2_qapair_fp']
    ]
    meta_fp_dist = datafiles['all_qapair_fp']

    # Load metadata from JSON files
    metadata = []
    for meta_fp in meta_fps:
        with open(meta_fp, "r") as f:
            metadata.extend(json.load(f))  # Assuming metadata is a list of {image_path, question, answer}

    with open(meta_fp_dist, "w") as f:
        json.dump(metadata, f)
