# BoundingBoxDrawer

This code was created to generate `.jpg` images with bounding box(es) (rectangle only) from directory that contains JSON files created by [LabelMe](https://github.com/LabelMe/labelme)

## Requirements
- Python 3.7+
- Install `requirements.txt`

        pip3 install -r requirements.txt

    OR

        pip install -r requirements.txt

## Usage

1. Open `draw_boxes.py`
2. Config following settings
    
    ```python
        # General setting 
        # ! Path should end with "/" if not empty "" ***
        # this directory must contain json of labelme data (original image is optional)
        PATH_TO_JSONS = "img/"
        PATH_TO_SAVE_IMG = "labeled_img/"  # path to save image with bounding box

       # Bounding box setting 
        BOX_COLOR = (0, 0, 255)  # (B,G,R)
        BOX_THICKNESS = 2  # in pixel
    ```
    > **NOTE:** Path in `PATH_TO_JSONS` and `PATH_TO_SAVE_IMG` should end with `/` e.g. `"path/to/json/"`, `"path/to/img/"` if not empty e.g. `""`   

    > **NOTE:** For `PATH_TO_SAVE_IMG` if directory specify in path does not exist, the code will automatically generate that directory.

3. Run the code in terminal or cmd

        python3 draw_boxes.py
    
    OR

        python draw_boxes.py