import json
import io
import os
import base64
import numpy as np
from PIL import Image
import cv2
from glob import glob
from pathlib import Path

# This code should work with Labelme JSON output

####################### General setting #######################
# this directory must contain json of labelme data (original image is optional)
PATH_TO_JSONS = "img"
PATH_TO_SAVE_IMG = "labeled_img"  # path to save image with bounding box

##################### Bounding box setting ####################
BOX_COLOR = (0, 255, 255)  # (B,G,R)
BOX_THICKNESS = 6  # in pixel

# decode base64 image to numpy array
def img_b64_to_arr(img_b64: str):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(Image.open(f))
    return img_arr

# get all box coordinates
def get_boxes_coor(shapes: list):
    boxes = []
    labels = []
    for box in shapes:
        if box["shape_type"] == "rectangle":
            x_start = round(box["points"][0][0])
            y_start = round(box["points"][0][1])
            x_end = round(box["points"][1][0])
            y_end = round(box["points"][1][1])
            boxes.append([(x_start, y_start), (x_end, y_end)])
            labels.append(box["label"])
    return boxes, labels

# draw all boxes on image (numpy array)
def draw_boxes(img_arr: np.ndarray, boxes_coor: list, labels: list):
    image_rect = img_arr
    for idx, box_coor in enumerate(boxes_coor):
        image_rect = cv2.rectangle(
            image_rect, box_coor[0], box_coor[1], BOX_COLOR, BOX_THICKNESS)
        cv2.putText(image_rect, labels[idx], (box_coor[0][0], box_coor[0][1]-30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, thickness=3, color=BOX_COLOR)
    return image_rect

def draw_boxes_on_img():
    # get all "path" to labelme JSONS files
    all_files = glob(os.path.join( PATH_TO_JSONS, "*.json"))

    # create save directory if not exist
    Path(PATH_TO_SAVE_IMG).mkdir(parents=True, exist_ok=True)

    # loop write all labeled images
    for this_file in all_files:
        # get file name from path
        file_name = os.path.basename(this_file).split(".")[0]

        # read all data from labelme JSONS file
        label_data = json.load(open(this_file))

        # get img from encoded data save in JSON
        img_arr = img_b64_to_arr(label_data['imageData'])

        # case image is grayscale (like from .bmp) convert to BGR first
        if img_arr.ndim == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)

        # get all box coordinates
        boxes_coor, labels = get_boxes_coor(label_data["shapes"])

        # draw boxes on image
        image = draw_boxes(img_arr, boxes_coor, labels)

        # save image
        cv2.imwrite(os.path.join(PATH_TO_SAVE_IMG, f"{file_name}.jpg"), image)


if __name__ == "__main__":
    draw_boxes_on_img()
