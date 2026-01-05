import cv2
import numpy as np
import os
import glob

# Constants
CARD_W = 90.0  # mm
CARD_H = 55.0  # mm
CARD_RATIO = CARD_W / CARD_H
ARUCO_SIZE_MM = 40.0
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARGIN = 10
MIN_CARD_AREA = 3000
CENTER_CONTOUR_COLOR = (0, 0, 255)
HIGHLIGHT_COLOR = (0, 255, 0)
HIGHLIGHT_ALPHA = 0.3

def auto_canny(gray, sigma=0.33):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def preprocess(img, blur_size=(5,5), thresh_val=50, canny_sigma=0.33, kernel_shape=(3,7), use_thresh=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blur_size, 0)
    if use_thresh:
        _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
        edge_input = thresh
    else:
        edge_input = blur
    edge = auto_canny(edge_input, sigma=canny_sigma)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
    return cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)

def get_objects(edge, min_area):
    cnts, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnts if cv2.contourArea(c) >= min_area]

def measure_pca(contour):
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean = np.mean(pts, axis=0)
    pts -= mean
    cov = np.cov(pts.T)
    eigval, eigvec = np.linalg.eig(cov)
    idx = eigval.argsort()[::-1]
    eigvec = eigvec[:, idx]
    proj = pts @ eigvec
    L = abs(proj[:, 0].max() - proj[:, 0].min())
    W = abs(proj[:, 1].max() - proj[:, 1].min())
    if W > L:
        L, W = W, L
    return L, W

def find_center_object(objects, img_shape):
    if not objects:
        return None
    center_img = (img_shape[1] // 2, img_shape[0] // 2)
    best, min_dist = None, float('inf')
    for c in objects:
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        dist = np.sqrt((cx - center_img[0])**2 + (cy - center_img[1])**2)
        if dist < min_dist:
            min_dist = dist
            best = c
    return best

def crop_with_margin(img, rect, margin=MARGIN):
    x, y, w, h = rect
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2 * margin)
    h = min(img.shape[0] - y, h + 2 * margin)
    return img[y:y+h, x:x+w]

def highlight_object(result, contour, color=HIGHLIGHT_COLOR, alpha=HIGHLIGHT_ALPHA):
    overlay = result.copy()
    cv2.fillConvexPoly(overlay, contour, color)
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    cv2.drawContours(result, [contour], -1, color, 2)

def detect_and_crop_aruco(img, result):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, None
    aruco_corners = corners[0][0].astype(int)
    highlight_object(result, aruco_corners)
    x_min, x_max = min(aruco_corners[:, 0]), max(aruco_corners[:, 0])
    y_min, y_max = min(aruco_corners[:, 1]), max(aruco_corners[:, 1])
    rect = (x_min, y_min, x_max - x_min, y_max - y_min)
    crop = crop_with_margin(img, rect, margin=5)
    px = np.linalg.norm(aruco_corners[0] - aruco_corners[1])
    ppm = px / ARUCO_SIZE_MM
    return crop, ppm

def find_card(img, result):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    adap_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edge = auto_canny(adap_thresh, sigma=0.2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_err = None, float('inf')
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_CARD_AREA:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) != 4:
            continue
        rect = cv2.minAreaRect(approx)
        w_px, h_px = rect[1]
        if w_px == 0 or h_px == 0:
            continue
        ratio = max(w_px, h_px) / min(w_px, h_px)
        err = abs(ratio - CARD_RATIO)
        if err < 0.15:
            best = approx.reshape(-1, 2).astype(int)
            best_err = err
    if best is not None:
        highlight_object(result, best)
    return best

def process_single_image(img_path, output_dir):
    """Xử lý một ảnh duy nhất và lưu kết quả."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot read image: {img_path}")
        return
    
    # Lấy tên file gốc (không có extension)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    result = img.copy()

    # Phát hiện scale: Ưu tiên card
    print(f"Processing {base_name}: Prioritize card detection...")
    card = find_card(img, result)
    if card is not None:
        rect = cv2.minAreaRect(card)
        w_px, h_px = rect[1]
        ppm_w = max(w_px, h_px) / CARD_W
        ppm_h = min(w_px, h_px) / CARD_H
        ppm = (ppm_w + ppm_h) / 2
        crop_rect = cv2.boundingRect(card)
        scale_crop = crop_with_margin(img, crop_rect)
        scale_path = os.path.join(output_dir, f"{base_name}_scale.jpg")
        cv2.imwrite(scale_path, scale_crop)
        print(f"Success: Scale from CARD | Average PPM = {ppm:.3f}")
    else:
        print(f"Warning: {base_name}: CARD not found, fallback to ArUco")
        aruco_crop, ppm = detect_and_crop_aruco(img, result)
        if aruco_crop is None:
            print(f"Warning: {base_name}: ArUco not found, using default ppm=1.0")
            ppm = 1.0
            scale_crop = None
        else:
            scale_crop = aruco_crop
            scale_path = os.path.join(output_dir, f"{base_name}_scale.jpg")
            cv2.imwrite(scale_path, scale_crop)
            print(f"Success: {base_name}: Scale from ARUCO | PPM = {ppm:.3f}")

    # Phát hiện vật thể
    edge = preprocess(img)
    objects = get_objects(edge, 1000)

    # Tìm và đo vật thể trung tâm
    center_contour = find_center_object(objects, img.shape)
    if center_contour is None:
        print(f"Error: {base_name}: No object found in center")
        return

    # Làm mịn contour
    peri = cv2.arcLength(center_contour, True)
    center_contour = cv2.approxPolyDP(center_contour, 0.005 * peri, True)

    L_px, W_px = measure_pca(center_contour)
    L_mm, W_mm = L_px / ppm, W_px / ppm
    print(f"Success: {base_name}: Center object size: Length = {L_mm:.2f} mm, Width = {W_mm:.2f} mm")

    cv2.drawContours(result, [center_contour], -1, CENTER_CONTOUR_COLOR, 2)
    M = cv2.moments(center_contour)
    if M['m00'] != 0:
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        text = f"L: {L_mm:.1f}mm, W: {W_mm:.1f}mm"
        cv2.putText(result, text, (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Lưu result
    result_path = os.path.join(output_dir, f"{base_name}_result.jpg")
    cv2.imwrite(result_path, result)

    # Crop và lưu center object
    crop_rect = cv2.boundingRect(center_contour)
    center_crop = crop_with_margin(img, crop_rect)
    center_path = os.path.join(output_dir, f"{base_name}_center.jpg")
    cv2.imwrite(center_path, center_crop)
    print(f"Success: {base_name}: Results saved to {output_dir}")

def main(input_dir, output_dir):
    """Hàm chính: Xử lý tất cả ảnh trong input_dir và xuất ra output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Info: Created output directory: {output_dir}")
    
    # Liệt kê tất cả file ảnh
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"Error: No images found in {input_dir}")
        return
    
    print(f"Info: Starting to process {len(image_files)} images...")
    for img_path in image_files:
        process_single_image(img_path, output_dir)
    print("Success: All images processed!")

if __name__ == "__main__":
    # Gán thư mục đầu vào và đầu ra
    input_dir = os.getcwd()  # Thư mục chứa ảnh đầu vào (thư mục hiện tại)
    output_dir = os.path.join(os.getcwd(), "result")  # Thư mục xuất kết quả (sẽ tự tạo nếu chưa có)
    main(input_dir, output_dir)