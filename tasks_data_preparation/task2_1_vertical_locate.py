import os
import cv2
import glob 
import json
import numpy as np
from pathlib import Path

def draw_horizontal_lines_and_blocks(img):
    """
    Draws 9 vertical red lines to separate an img into 10 horizontal blocks, 
    numbers them from 0 to 9, and saves the resulting img to the destination path.

    Parameters:
        img_fp_src (str): Source img file path.
        img_fp_dst (str): Destination img file path.
    """
    # Get img dimensions
    height, width = img.shape[:2]

    # Calculate block height
    block_height = height // 10

    # Font and text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)  # White

    # Draw 9 vertical lines and add text
    for i in range(10):
        if i < 9:  # Draw vertical lines (exclude the last boundary line)
            y = (i + 1) * block_height
            cv2.line(img, (0, y), (width, y), (0, 0, 255), 2)  # Red line

        # Calculate text position
        text = str(i)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (width - text_size[0]) // 2  # Center horizontally
        text_y = (i * block_height + (block_height + text_size[1]) // 2)  # Center vertically

        # Put the text on the img
        cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    return img

if __name__ == "__main__":
    with open("datafiles.json", "r") as f:
        datafiles = json.load(f)['preview']

    for datafile in datafiles:        
        img_dir = datafile['img_dir']
        task2_img_dir = datafile['task2_1_img_dir']
        Path(task2_img_dir).mkdir(exist_ok=True, parents=True)

        img_fps = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        for img_fp_src in img_fps[:10]:
            img_fn = os.path.basename(img_fp_src)
            img_fp_dst  = os.path.join(task2_img_dir, img_fn)

            # Load the image
            image = cv2.imread(img_fp_src)
            if image is None:
                raise FileNotFoundError(f"The file at path {img_fp_src} was not found.")
            
            image = draw_horizontal_lines_and_blocks(image)

            # Save the modified image
            cv2.imwrite(img_fp_dst, image)




