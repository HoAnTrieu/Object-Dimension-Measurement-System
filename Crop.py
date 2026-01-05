from PIL import Image
import os

# Thư mục chứa ảnh (đổi thành đường dẫn của bạn)
input_folder = "720x960"
output_folder = "images_cropped"

# Tạo thư mục output nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Duyệt tất cả file trong thư mục
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with Image.open(input_path) as img:
            width, height = img.size

            # Kiểm tra size để tránh lỗi
            if width == 720 and height >= 845:
                cropped_img = img.crop((0, 0, 720, 845))
                cropped_img.save(output_path)
                print(f"Đã crop: {filename}")
            else:
                print(f"Bỏ qua (không đúng size): {filename}")

print("Hoàn tất!")
