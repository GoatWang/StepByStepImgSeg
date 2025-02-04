import json

def read_labelme_annotation(json_path):
    """Reads a LabelMe annotation JSON file and prints out the shapes and labels."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # for shape in data.get("shapes", []):
    #     label = shape.get("label", "Unknown")
    #     shape_type = shape.get("shape_type", "Unknown")
    #     points = shape.get("points", [])

    return data

# def 


if __name__ == "__main__":
    import os
    import json
    import glob
    

    with open("datafiles.json", "r") as f:
        datafiles = json.load(f)

    for datafile in datafiles:
        img_dir = datafile['imgmask_filtered_dir']
        mask_dir = datafile['task3_samedist_gap_dir']

        img_fps = glob.glob(os.path.join(img_dir, "*.jpg"))
        get_mask_fp = lambda img_fp: os.path.join(mask_dir, os.path.basename(img_fp).replace(".jpg", ".json"))
        mask_fn = os.path.basename(img_fps[0]).replace(".jpg", ".json")
        img_mask_fp_pairs = [(img_fp, get_mask_fp(img_fp)) for img_fp in img_fps]
        
        for img_fp, mask_fp in img_mask_fp_pairs[:10]:
            anno_data = read_labelme_annotation(mask_fp)
            for  shape in anno_data['shapes']:
                label = shape['label']
                points = shape['points']
                
                for i in range(len(points)):
                    points_temp = points[0]

                # 1. draw the previous poiuts
                # 2. draw clock ticks
                # 3. put text on each portion
                # 4. produce question file







