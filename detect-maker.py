import cv2
import cv2.aruco as aruco
import numpy as np
import os

# ---------- LOAD IMAGE ----------
IMAGE_PATH = r"img_7000.jpg"

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError("Image path not found")

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("Cannot load image")

# ---------- PREPROCESSING (CỨU MỜ NHẸ) ----------
# 1. Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. CLAHE – tăng tương phản
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray_clahe = clahe.apply(gray)

# 3. Gaussian Blur nhẹ – giảm nhiễu
gray_blur = cv2.GaussianBlur(gray_clahe, (3, 3), 0)

# 4. Unsharp Mask – làm sắc cạnh
blur_for_sharp = cv2.GaussianBlur(gray_blur, (0, 0), sigmaX=1.0)
gray_sharp = cv2.addWeighted(
    gray_blur, 1.5,
    blur_for_sharp, -0.5,
    0
)

# ---------- ARUCO SETUP ----------
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

params = aruco.DetectorParameters()
params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 23
params.adaptiveThreshWinSizeStep = 10
params.minMarkerPerimeterRate = 0.03
params.maxMarkerPerimeterRate = 4.0

detector = aruco.ArucoDetector(aruco_dict, params)

# ---------- DETECT ----------
corners, ids, rejected = detector.detectMarkers(gray_sharp)

if ids is not None:
    aruco.drawDetectedMarkers(img, corners, ids)
    print("Detected ArUco IDs:", ids.flatten())
else:
    print("No ArUco marker detected")

# ---------- RESIZE FOR DISPLAY ----------
max_width = 900
h, w = img.shape[:2]

if w > max_width:
    scale = max_width / w
    display_img = cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA
    )
else:
    display_img = img.copy()

# ---------- SHOW ----------
cv2.imshow("Detected ArUco (Software Enhanced)", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
