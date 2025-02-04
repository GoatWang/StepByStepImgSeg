# filter the image with only one object for each class
import os
import json
import shutil
from PIL import Image

def filter_training_data(img_dir, mask_dir, imgmask_filtered_dir):
    if not os.path.exists(imgmask_filtered_dir):
        os.makedirs(imgmask_filtered_dir)

    for anno_file in os.listdir(mask_dir):
        if not anno_file.endswith('.json'):
            continue

        # Load annotation file
        anno_path = os.path.join(mask_dir, anno_file)
        with open(anno_path, 'r') as f:
            anno_data = json.load(f)

        # Filter out labels with multiple same-class objects
        label_counts = {}
        filtered_shapes = []

        for shape in anno_data['shapes']:
            label = shape['label']
            label_counts[label] = label_counts.get(label, 0) + 1

        for shape in anno_data['shapes']:
            if label_counts[shape['label']] == 1:
                filtered_shapes.append(shape)

        # Calculate image size and filter by bbox dimensions
        img_path = os.path.join(img_dir, anno_data['imagePath'])
        if not os.path.exists(img_path):
            continue

        with Image.open(img_path) as img:
            img_width, img_height = img.size

        valid_shapes = []

        for shape in filtered_shapes:
            points = shape['points']
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            bbox_width = max(x_coords) - min(x_coords)
            bbox_height = max(y_coords) - min(y_coords)

            if bbox_width >= img_width / 10 and bbox_height >= img_height / 10:
                valid_shapes.append(shape)

        # If there are valid shapes left, save the image and annotation
        if valid_shapes:
            # Update annotation data
            anno_data['shapes'] = valid_shapes

            # Save the filtered annotation
            filtered_anno_path = os.path.join(imgmask_filtered_dir, anno_file)
            with open(filtered_anno_path, 'w') as f:
                json.dump(anno_data, f, indent=4)

            # Copy the image
            filtered_img_path = os.path.join(imgmask_filtered_dir, anno_data['imagePath'])
            shutil.copy(img_path, filtered_img_path)

# Example usage:
# filter_training_data('path/to/img_dir', 'path/to/mask_dir', 'path/to/imgmask_filtered_dir')
if __name__ == "__main__":
    from pprint import pprint
    with open("datafiles.json", "r") as f:
        datafiles = json.load(f)

    for datafile in datafiles:
        img_dir = datafile["img_dir"]
        mask_dir = datafile["mask_dir"]
        imgmask_filtered_dir = datafile["imgmask_filtered_dir"]
        filter_training_data(img_dir, mask_dir, imgmask_filtered_dir)


