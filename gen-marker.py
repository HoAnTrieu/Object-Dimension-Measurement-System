import cv2
import cv2.aruco as aruco
import numpy as np

# Chọn dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

marker_id = 0
marker_size_px = 500  # kích thước ảnh marker (pixel)

# Tạo marker
marker_img = aruco.generateImageMarker(
    aruco_dict,
    marker_id,
    marker_size_px
)

# Lưu file
cv2.imwrite("aruco_marker_id0.png", marker_img)

# Hiển thị
cv2.imshow("ArUco Marker", marker_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
