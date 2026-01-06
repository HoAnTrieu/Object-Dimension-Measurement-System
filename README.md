# Image Processing Scripts

This repository contains Python scripts for image processing tasks including cropping, resizing, ArUco marker detection, and generation.

## Prerequisites

- Python 3.x installed
- Install required libraries:
  ```
  pip install pillow opencv-python numpy
  ```

## Scripts

### Crop.py

Crops images to a specific size (720x845 pixels) if they match width 720 and height >=845.

#### Setup
- Create a folder named `720x960` in the same directory as the script.
- Place images (.jpg, .jpeg, .png, .webp) in the `720x960` folder.

#### Usage
Run the script:
```
python Crop.py
```
It will create an `images_cropped` folder and save cropped images there. Skips images that don't match the size criteria.

### detect-maker.py

Detects ArUco markers in a single image using OpenCV.

#### Setup
- Place an image file named `img_7000.jpg` in the same directory as the script.

#### Usage
Run the script:
```
python detect-maker.py
```
It will process the image, detect markers, draw them, and display the result in a window. Press any key to close.

### resize.py

Resizes images to 720x960 pixels without maintaining aspect ratio.

#### Setup
- Create directories:
  - `C:\Users\HAT\Desktop\XLAS DATA\DATA\input` (place input images here)
  - The script will create `C:\Users\HAT\Desktop\XLAS DATA\DATA\output` automatically.
- Place images (.jpg, .jpeg, .png, .bmp) in the input directory.

#### Usage
Run the script:
```
python resize.py
```
It will resize all images, rename them to `img_01.jpg`, `img_02.jpg`, etc., and save in the output directory.

### gen-marker.py

Generates an ArUco marker image.

#### Setup
- No additional setup required.

#### Usage
Run the script:
```
python gen-marker.py
```
It will generate a 500x500 pixel ArUco marker (ID 0), save it as `aruco_marker_id0.png`, and display it in a window. Press any key to close.

### main.py

Processes images in the current directory to detect and measure the center object using edge detection and contour analysis. Uses ArUco markers or predefined card dimensions for scaling to convert measurements to mm. Saves highlighted results, cropped images, and scale references to a "result" subfolder.

#### Setup
- Place .jpg, .jpeg, or .png images in the same directory as the script.

#### Usage
Run the script:
```
python main.py
```
It will process all images, output measurements to console, and save results to the "result" folder.

## How to Use and Workflow

The scripts form a sequential pipeline for image processing and object measurement. Start with preparation, then preprocessing, and finally detection/measurement.

1. **Generate Markers**: Run `gen-marker.py` to create ArUco markers for scaling reference. Print or place them in images.
2. **Prepare Images**: Use `resize.py` or `Crop.py` to standardize image sizes. Place images in the required input folders (e.g., `720x960` for Crop.py, or hardcoded paths for resize.py).
3. **Preprocess Images**: Run `TienXuLy.py` via Streamlit (`streamlit run TienXuLy.py`) to interactively apply preprocessing (blur, edges, thresholds, masks). Adjust parameters in the sidebar, process images from a `data` folder, and review outputs in `output` subfolders. This refines images for better downstream accuracy.
4. **Detect and Measure**: Run `main.py` on the (preprocessed) images to detect center objects, measure dimensions in mm (using markers/cards for scaling), and save annotated results to `result`. Use `detect-maker.py` to test marker detection on a sample image first.
5. **Utilities**: Run scripts individually as needed. Ensure dependencies are installed (`pip install pillow opencv-python numpy streamlit`).

Workflow order: Preparation → Preprocessing (TienXuLy.py) → Measurement (main.py). Outputs from one can feed into another (e.g., preprocessed masks improve main.py detection).

## Project Members
- Ho An Trieu
- Lai Do Minh Quan
- Nguyen Vu Thanh

## Notes

- Adjust hardcoded paths in the scripts as needed for your environment.
- Ensure write permissions for output directories.
- For detect-maker.py and gen-marker.py, OpenCV must be compiled with ArUco support.