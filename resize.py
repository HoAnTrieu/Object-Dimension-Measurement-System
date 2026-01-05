import cv2
import os

# --------- CẤU HÌNH ----------
INPUT_DIR = r"C:\Users\HAT\Desktop\XLAS DATA\DATA\input"
OUTPUT_DIR = r"C:\Users\HAT\Desktop\XLAS DATA\DATA\output"

TARGET_WIDTH = 720
TARGET_HEIGHT = 960

# Tạo thư mục output nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lấy danh sách file ảnh (lọc theo đuôi)
valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
image_files = sorted(
    [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_ext)]
)

if not image_files:
    raise RuntimeError("No image files found in input directory")

# --------- XỬ LÝ ----------
for idx, filename in enumerate(image_files, start=1):
    input_path = os.path.join(INPUT_DIR, filename)

    img = cv2.imread(input_path)
    if img is None:
        print(f"Skip unreadable file: {filename}")
        continue

    # Resize (không giữ tỉ lệ – đúng yêu cầu)
    resized = cv2.resize(
        img,
        (TARGET_WIDTH, TARGET_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # Đặt tên mới
    new_name = f"img_{idx:02d}.jpg"
    output_path = os.path.join(OUTPUT_DIR, new_name)

    cv2.imwrite(output_path, resized)
    print(f"Saved: {new_name}")

print("Done resizing and renaming images.")
