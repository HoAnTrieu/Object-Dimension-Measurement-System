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

## Cách sử dụng và Quy trình làm việc

Các tập lệnh tạo thành một quy trình tuần tự để xử lý hình ảnh và đo kích thước vật thể. Bắt đầu với chuẩn bị, sau đó tiền xử lý, và cuối cùng là phát hiện/đo lường.

1. **Tạo mã đánh dấu**: Chạy `gen-marker.py` để tạo mã đánh dấu ArUco để tham chiếu tỷ lệ. In hoặc đặt chúng vào hình ảnh.
2. **Chuẩn bị hình ảnh**: Sử dụng `resize.py` hoặc `Crop.py` để chuẩn hóa kích thước hình ảnh. Đặt hình ảnh vào thư mục đầu vào cần thiết (ví dụ: `720x960` cho Crop.py, hoặc đường dẫn được mã hóa cứng cho resize.py).
3. **Tiền xử lý hình ảnh**: Chạy `TienXuLy.py` qua Streamlit (`streamlit run TienXuLy.py`) để tương tác áp dụng tiền xử lý (làm mờ, cạnh, ngưỡng, mặt nạ). Điều chỉnh tham số trong thanh bên, xử lý hình ảnh từ thư mục `data`, và xem xét đầu ra trong thư mục con `output`. Điều này tinh chỉnh hình ảnh để cải thiện độ chính xác ở hạ nguồn.
4. **Phát hiện và đo lường**: Chạy `main.py` trên hình ảnh (đã tiền xử lý) để phát hiện vật thể trung tâm, đo kích thước theo mm (sử dụng mã đánh dấu/thẻ để tỷ lệ), và lưu kết quả chú thích vào `result`. Sử dụng `detect-maker.py` để kiểm tra phát hiện mã đánh dấu trên hình ảnh mẫu trước.
5. **Tiện ích**: Chạy tập lệnh riêng lẻ khi cần. Đảm bảo cài đặt phụ thuộc (`pip install pillow opencv-python numpy streamlit`).

Thứ tự quy trình: Chuẩn bị → Tiền xử lý (TienXuLy.py) → Đo lường (main.py). Đầu ra từ một tập lệnh có thể cấp cho tập lệnh khác (ví dụ: mặt nạ tiền xử lý cải thiện phát hiện main.py).

## Thành viên dự án
- Ho An Trieu
- Lai Do Minh Quan
- Nguyen Vu Thanh

## Lưu ý

- Điều chỉnh các đường dẫn được mã hóa cứng trong tập lệnh nếu cần cho môi trường của bạn.
- Đảm bảo quyền ghi cho các thư mục đầu ra.
- Đối với detect-maker.py và gen-marker.py, OpenCV phải được biên dịch với hỗ trợ ArUco.