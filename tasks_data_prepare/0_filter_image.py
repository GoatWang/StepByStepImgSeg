# filter the image with only one object for each class
import os
import json
import shutil
from PIL import Image

def filter_training_data(img_dir, anno_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for anno_file in os.listdir(anno_dir):
        if not anno_file.endswith('.json'):
            continue

        # Load annotation file
        anno_path = os.path.join(anno_dir, anno_file)
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
            filtered_anno_path = os.path.join(save_dir, anno_file)
            with open(filtered_anno_path, 'w') as f:
                json.dump(anno_data, f, indent=4)

            # Copy the image
            filtered_img_path = os.path.join(save_dir, anno_data['imagePath'])
            shutil.copy(img_path, filtered_img_path)

# Example usage:
# filter_training_data('path/to/img_dir', 'path/to/anno_dir', 'path/to/save_dir')
if __name__ == "__main__":
    img_dir = "../coco_dataset/val2017"
    anno_dir = "../coco_dataset/val2017_0_mask"
    save_dir = "../coco_dataset/val2017_1_imgmask_filtered"
    filter_training_data(img_dir, anno_dir, save_dir)


