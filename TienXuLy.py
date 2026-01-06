import streamlit as st
import cv2
import numpy as np
import os
from pathlib import Path

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Image Processing Pipeline", layout="wide")

DATA_DIR = "data"
CANNY_DIR = "output/canny"
BINARY_DIR = "output/binary"
MASK_DIR = "output/mask"
CANNY_MASK_DIR = "output/canny_mask"

os.makedirs(CANNY_DIR, exist_ok=True)
os.makedirs(BINARY_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(CANNY_MASK_DIR, exist_ok=True)

# =============================
# SIDEBAR
# =============================
st.sidebar.title("Cấu hình xử lý ảnh")

blur_kernel = st.sidebar.slider("Gaussian Blur", 3, 11, 5, step=2)

use_otsu = st.sidebar.checkbox("Dùng OTSU Threshold (khuyến nghị)", True)
manual_thresh = st.sidebar.slider("Threshold tay", 0, 255, 120)

kernel_size = st.sidebar.slider("Kernel Morphology", 3, 11, 5, step=2)
close_iter = st.sidebar.slider("Close (vá hở)", 1, 5, 2)
open_iter = st.sidebar.slider("Open (lọc nhiễu)", 1, 5, 1)

use_auto_canny = st.sidebar.checkbox("Auto Canny (tham khảo)", True)
canny_low = st.sidebar.slider("Canny Low", 0, 255, 60)
canny_high = st.sidebar.slider("Canny High", 0, 255, 150)

mask_source = st.sidebar.radio("Nguồn tạo MASK", ("Binary (Otsu/Manual)", "Canny-based", "So sánh cả hai"))

run_btn = st.sidebar.button("🚀 XỬ LÝ ẢNH")

# =============================
# FUNCTIONS
# =============================
def auto_canny(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper)
    return edges, lower, upper

def canny_to_mask(edges, kernel_size=5, close_iter=2, open_iter=1, min_area_ratio=0.0005):
    """
    Chuyển edge mảnh thành mask kín:
    1) Dilation/closing để nối cạnh,
    2) findContours + fill contour lớn,
    3) morphology open để loại nhiễu còn lại.
    min_area_ratio: ngưỡng bỏ contour quá nhỏ (so với diện tích ảnh).
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Tăng dày cạnh để dễ đóng và tìm contour
    dilated = cv2.dilate(edges, kernel, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    # Tìm contour trên ảnh nhị phân (closed)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = edges.shape[:2]
    img_area = h * w
    mask = np.zeros_like(edges)

    # Lọc contour nhỏ và vẽ fill
    min_area = max(1, int(min_area_ratio * img_area))
    big_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if big_contours:
        cv2.drawContours(mask, big_contours, -1, 255, thickness=cv2.FILLED)

    # Làm sạch bằng open
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    # Đảm bảo binary {0,255}
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def process_image(image, filename):
    result = {}

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result['gray'] = gray

    # 2. Gaussian Blur
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    result['blur'] = blur

    # 3. Canny Edge (CHỈ THAM KHẢO / CÓ THỂ DÙNG LÀM MASK)
    if use_auto_canny:
        edges, lower, upper = auto_canny(blur)
        result['canny_info'] = (lower, upper)
    else:
        edges = cv2.Canny(blur, canny_low, canny_high)
        result['canny_info'] = (canny_low, canny_high)
    result['canny'] = edges
    cv2.imwrite(os.path.join(CANNY_DIR, filename), edges)

    # 4. Binary Threshold (SINH VÙNG KÍN)
    if use_otsu:
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blur, manual_thresh, 255, cv2.THRESH_BINARY)

    # Đảm bảo: vật trắng – nền đen
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    result['binary'] = binary
    cv2.imwrite(os.path.join(BINARY_DIR, filename), binary)

    # 5. Morphology trên binary để ra mask chuẩn
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    mask_binary = opened
    result['mask_binary'] = mask_binary
    cv2.imwrite(os.path.join(MASK_DIR, filename), mask_binary)

    # 6. Tạo mask từ canny (nếu cần)
    mask_from_canny = canny_to_mask(edges, kernel_size=kernel_size, close_iter=close_iter, open_iter=open_iter)
    result['mask_canny'] = mask_from_canny
    cv2.imwrite(os.path.join(CANNY_MASK_DIR, filename), mask_from_canny)

    return result

def compute_metrics(mask_a, mask_b):
    """
    Các metric đơn giản giữa hai mask nhị phân (0/255): IoU, Dice, số CC, area diff
    Trả về dict kết quả.
    """
    a = (mask_a > 0)
    b = (mask_b > 0)

    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    area_a = np.count_nonzero(a)
    area_b = np.count_nonzero(b)

    iou = intersection / union if union > 0 else 1.0
    dice = (2 * intersection) / (area_a + area_b) if (area_a + area_b) > 0 else 1.0
    area_rel_diff = (area_b - area_a) / area_a if area_a > 0 else float('inf')

    # số connected components (không tính background)
    n_labels_a, _, _, _ = cv2.connectedComponentsWithStats(mask_a.astype('uint8'))
    n_labels_b, _, _, _ = cv2.connectedComponentsWithStats(mask_b.astype('uint8'))

    stats = {
        'IoU': float(iou),
        'Dice': float(dice),
        'Area_mask_A_px': int(area_a),
        'Area_mask_B_px': int(area_b),
        'Area_rel_diff': float(area_rel_diff),
        'Connected_components_A': int(max(0, n_labels_a - 1)),
        'Connected_components_B': int(max(0, n_labels_b - 1)),
    }
    return stats

# =============================
# MAIN
# =============================
st.title("Pipeline xử lý ảnh số — Hỗ trợ Canny → Mask và So sánh")

st.markdown("""
**Ảnh gốc ⇒ Grayscale ⇒ Gaussian ⇒ Canny (tham khảo hoặc dùng để tạo mask) ⇒ Binary Threshold ⇒ MASK**

- Bạn có thể chọn `Nguồn tạo MASK` để so sánh kết quả.
""")

image_files = list(Path(DATA_DIR).glob("*.*"))

if not image_files:
    st.warning("Không có ảnh trong thư mục data/")
    st.stop()

if run_btn:
    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = process_image(image, img_path.name)

        st.markdown(f"## 📷 {img_path.name}")

        cols = st.columns([2,0.2,2,0.2,2,0.2,2,0.2,2,0.2,2])
        cols[0].image(image, channels="BGR", caption="Ảnh gốc")
        cols[2].image(result['gray'], caption="Grayscale")
        cols[4].image(result['blur'], caption="Gaussian Blur")
        cols[6].image(result['canny'], caption="Canny Edge (raw)", clamp=True)
        cols[8].image(result['binary'], caption="Binary Threshold")
        cols[10].image(result['mask_binary'], caption="MASK (từ Binary)")

        # Nếu chọn Canny-based hoặc compare, hiển thị mask từ Canny
        if mask_source in ("Canny-based", "So sánh cả hai"):
            st.image(result['mask_canny'], caption="MASK (từ Canny → fill)")

        # Nếu so sánh, tính metric giữa mask_binary và mask_canny
        if mask_source == "So sánh cả hai":
            stats = compute_metrics(result['mask_binary'], result['mask_canny'])
            st.markdown("### So sánh quantitative")
            st.write(stats)

        st.divider()

    st.success(
        f"Hoàn tất xử lý {len(image_files)} ảnh\n\n"
        f"- Canny raw: `{CANNY_DIR}`\n"
        f"- Binary: `{BINARY_DIR}`\n"
        f"- Mask binary: `{MASK_DIR}`\n"
        f"- Mask from Canny: `{CANNY_MASK_DIR}`"
    )
else:
    st.info("Điều chỉnh tham số bên trái và nhấn **XỬ LÝ ẢNH**")
