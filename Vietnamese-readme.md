# Tập lệnh xử lý hình ảnh

Kho lưu trữ này chứa các tập lệnh Python để xử lý hình ảnh bao gồm cắt, thay đổi kích thước, phát hiện và tạo mã đánh dấu ArUco.

## Điều kiện tiên quyết

- Python 3.x đã được cài đặt
- Cài đặt các thư viện cần thiết:
  ```
  pip install pillow opencv-python numpy
  ```

## Tập lệnh

### Crop.py

Cắt hình ảnh thành kích thước cụ thể (720x845 pixel) nếu chúng khớp chiều rộng 720 và chiều cao >=845.

#### Thiết lập
- Tạo một thư mục có tên `720x960` trong cùng thư mục với tập lệnh.
- Đặt hình ảnh (.jpg, .jpeg, .png, .webp) vào thư mục `720x960`.

#### Cách sử dụng
Chạy tập lệnh:
```
python Crop.py
```
Nó sẽ tạo thư mục `images_cropped` và lưu hình ảnh đã cắt ở đó. Bỏ qua hình ảnh không khớp với tiêu chí kích thước.

### detect-maker.py

Phát hiện mã đánh dấu ArUco trong một hình ảnh duy nhất bằng OpenCV.

#### Thiết lập
- Đặt tệp hình ảnh có tên `img_7000.jpg` trong cùng thư mục với tập lệnh.

#### Cách sử dụng
Chạy tập lệnh:
```
python detect-maker.py
```
Nó sẽ xử lý hình ảnh, phát hiện mã đánh dấu, vẽ chúng và hiển thị kết quả trong cửa sổ. Nhấn bất kỳ phím nào để đóng.

### resize.py

Thay đổi kích thước hình ảnh thành 720x960 pixel mà không giữ tỷ lệ khung hình.

#### Thiết lập
- Tạo các thư mục:
  - `C:\Users\HAT\Desktop\XLAS DATA\DATA\input` (đặt hình ảnh đầu vào ở đây)
  - Tập lệnh sẽ tự động tạo `C:\Users\HAT\Desktop\XLAS DATA\DATA\output`.
- Đặt hình ảnh (.jpg, .jpeg, .png, .bmp) vào thư mục đầu vào.

#### Cách sử dụng
Chạy tập lệnh:
```
python resize.py
```
Nó sẽ thay đổi kích thước tất cả hình ảnh, đổi tên chúng thành `img_01.jpg`, `img_02.jpg`, v.v., và lưu trong thư mục đầu ra.

### gen-marker.py

Tạo hình ảnh mã đánh dấu ArUco.

#### Thiết lập
- Không cần thiết lập bổ sung.

#### Cách sử dụng
Chạy tập lệnh:
```
python gen-marker.py
```
Nó sẽ tạo mã đánh dấu ArUco 500x500 pixel (ID 0), lưu dưới dạng `aruco_marker_id0.png`, và hiển thị trong cửa sổ. Nhấn bất kỳ phím nào để đóng.

### main.py

Xử lý hình ảnh trong thư mục hiện tại để phát hiện và đo kích thước vật thể ở giữa bằng cách sử dụng phát hiện cạnh và phân tích đường viền. Sử dụng mã đánh dấu ArUco hoặc kích thước thẻ được định nghĩa trước để chia tỷ lệ và chuyển đổi phép đo thành mm. Lưu kết quả được đánh dấu, hình ảnh được cắt và tham chiếu tỷ lệ vào thư mục con "result".

#### Thiết lập
- Đặt hình ảnh .jpg, .jpeg hoặc .png vào cùng thư mục với tập lệnh.

#### Cách sử dụng
Chạy tập lệnh:
```
python main.py
```
Nó sẽ xử lý tất cả hình ảnh, xuất phép đo ra bảng điều khiển và lưu kết quả vào thư mục "result".

## Lưu ý

- Điều chỉnh các đường dẫn được mã hóa cứng trong tập lệnh nếu cần cho môi trường của bạn.
- Đảm bảo quyền ghi cho các thư mục đầu ra.
- Đối với detect-maker.py và gen-marker.py, OpenCV phải được biên dịch với hỗ trợ ArUco.