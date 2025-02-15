import os
import cv2
import json
import base64
import numpy as np
from pycocotools.coco import COCO

def convert_coco_to_labelme(coco_annotation_file, output_dir, coco_images_dir=None):
    """
    Converts COCO annotations into LabelMe format without image base64 encoding.
    
    Args:
        coco_annotation_file (str): Path to the COCO annotation JSON file.
        output_dir (str): Directory where the LabelMe JSON files will be saved.
        coco_images_dir (str): Path to the directory containing COCO images (optional, for reference). encode image into base 64.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load COCO annotations
    coco = COCO(coco_annotation_file)

    # Loop through each image
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs([img_id])[0]
        img_name = img_info["file_name"]
        img_width, img_height = img_info["width"], img_info["height"]

        # Get annotations for the current image
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)


        # Prepare the LabelMe format JSON
        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": img_name,
            "imageData": None,
            "imageHeight": img_height,
            "imageWidth": img_width,
        }

        # Encode image in base64
        if coco_images_dir:
            img_path = os.path.join(coco_images_dir, img_name)
            with open(img_path, "rb") as img_file:
                labelme_data['imageData'] = base64.b64encode(img_file.read()).decode("utf-8")

        # Add shapes (annotations) to the LabelMe JSON
        for ann in anns:
            if "segmentation" in ann and ann["segmentation"]:  # Ensure segmentation exists
                # Handle polygons
                if isinstance(ann["segmentation"], list):  # Polygon format
                    for seg in ann["segmentation"]:
                        labelme_data["shapes"].append({
                            "label": coco.loadCats([ann["category_id"]])[0]["name"],
                            "points": [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)],
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                        })
                # Handle RLE format (convert to polygons)
                elif isinstance(ann["segmentation"], dict):
                    mask = coco.annToMask(ann)
                    mask = np.ascontiguousarray(mask)
                    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # ondemand server (opencv 3.x)
                    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # not ondemand server
                    for contour in contours:
                        contour = contour.squeeze(1).tolist()
                        labelme_data["shapes"].append({
                            "label": coco.loadCats([ann["category_id"]])[0]["name"],
                            "points": contour,
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                        })

        # Save the LabelMe JSON for the current image
        labelme_json_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".json")
        with open(labelme_json_path, "w") as f:
            json.dump(labelme_data, f, indent=4)

        print(f"Saved: {labelme_json_path}")

def main():
    with open(os.path.join(os.path.dirname(__file__), "datafiles.json"), "r") as f:
        datafiles = json.load(f)['production']

    for datafile in datafiles:
        coco_anno_fp = datafile['coco_anno_fp']
        img_dir = datafile['img_dir']
        mask_dir = datafile["mask_dir"]
        convert_coco_to_labelme(coco_anno_fp, mask_dir, img_dir) # remove coco_images_dir to disable base64 encoding

if __name__ == "__main__":
    main()