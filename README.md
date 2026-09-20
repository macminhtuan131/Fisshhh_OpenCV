# Bộ lời giải bài tập Xử lý ảnh — chương 2 đến chương 6

Thư mục này gồm lời giải chi tiết và mã Python chạy được cho toàn bộ bài tập trong các PDF:

- [Chương 2 — Image Basics and Filtering](LOI_GIAI/giai_toan/chuong2.md)
- [Chương 3 — Edge Detection](LOI_GIAI/giai_toan/chuong3.md)
- [Chương 4 — Feature Detection](LOI_GIAI/giai_toan/chuong4.md)
- [Chương 5 — Segmentation](LOI_GIAI/giai_toan/chuong5.md)
- [Chương 6 — Deep Learning/MNIST](LOI_GIAI/giai_toan/chuong6.md)

Mã nguồn:

- `code/chuong2.py`: đọc ảnh, đổi màu, nội suy, histogram, quay ảnh, lọc ảnh.
- `code/chuong3.py`: đạo hàm, Sobel, Canny viết từng bước và so sánh OpenCV.
- `code/chuong4.py`: Harris và HOG viết từ công thức.
- `code/chuong5.py`: K-means, phân ngưỡng, phân cụm màu và hình thái học.
- `code/chuong6_mnist.py`: đọc IDX, xây dựng/huấn luyện/đánh giá/lưu/triển khai ANN MNIST.

## Cách học với bộ lời giải

Với mỗi bài, làm theo thứ tự:

1. Nhìn **dữ kiện** và **đại lượng cần tìm**.
2. Nhận dạng dạng bài qua mục **Mấu chốt chọn công thức**.
3. Tự tính một phần tử hoặc viết giả mã trước.
4. Đối chiếu phần **Các bước tính** và **Kết quả**.
5. Chạy mã, thay dữ liệu hoặc tham số để xem kết quả đổi như thế nào.

Ba quy ước xuyên suốt:

- Ma trận ảnh truy cập theo `image[y, x]` = `image[hàng, cột]`.
- OpenCV đọc ảnh màu theo BGR; Matplotlib hiển thị RGB.
- Các PDF gọi phép lọc là **tích chập**. Tích chập phải lật kernel 180°. `cv2.filter2D` không lật kernel (nó tính tương quan), nên mã có lật kernel khi cần khớp đúng bài tính tay.

## Cài môi trường cho chương 2–5

Từ PowerShell, tại thư mục `LOI_GIAI`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy opencv-python matplotlib
```

Chạy một chương, ví dụ:

```powershell
python .\code\chuong2.py
python .\code\chuong3.py
python .\code\chuong4.py
python .\code\chuong5.py
```

Kết quả được lưu vào `LOI_GIAI/output/chuongX`. Dùng `--show` nếu muốn đồng thời mở cửa sổ đồ thị:

```powershell
python .\code\chuong3.py --show
```

## Môi trường riêng cho chương 6

TensorFlow hiện không hỗ trợ mọi bản Python mới. Nên dùng Python 3.10 hoặc 3.11 trong môi trường riêng:

```powershell
conda create -n mnist python=3.10 -y
conda activate mnist
python -m pip install tensorflow opencv-python matplotlib numpy
python .\code\chuong6_mnist.py --mode train --epochs 10
```

Chạy nhanh để kiểm tra pipeline trước khi huấn luyện đầy đủ:

```powershell
python .\code\chuong6_mnist.py --mode train --epochs 1 --train-limit 5000 --test-limit 1000
```

Triển khai mô hình đã lưu trên ảnh viết tay có sẵn:

```powershell
python .\code\chuong6_mnist.py --mode predict --image "..\Bài tập đưa sinh viên\Chuong6\data\data_test\a2.png"
```

## Lưu ý về kết quả số

Một số slide làm tròn tọa độ nội suy thành `0.33` và `0.67` trước khi tính; nếu dùng chính xác `1/3` và `2/3`, vài ô có thể lệch 1 mức xám. Lời giải ghi cả quy tắc và kết quả theo slide. Trong thực tế nên giữ số thực đầy đủ đến phép gán ảnh cuối cùng.

## Ký hiệu Python nền tảng dùng trong các bài

| Ký hiệu/lệnh | Cách đọc và tác dụng |
|---|---|
| `x = 5` | Gán giá trị 5 cho biến `x` |
| `x == 5`, `x != 5` | So sánh bằng, so sánh khác |
| `a + b`, `a - b`, `a * b`, `a / b` | Cộng, trừ, nhân, chia |
| `x ** 2` | Bình phương `x` |
| `A @ B` | Nhân ma trận |
| `image[y, x]` | Pixel hàng `y`, cột `x` |
| `image[:, :, 0]` | Mọi hàng, mọi cột, kênh số 0 |
| `a:b` | Từ vị trí `a` đến trước vị trí `b` |
| `()` | Gọi hàm hoặc gom biểu thức |
| `[]` | Danh sách hoặc truy cập mảng |
| `:` | Bắt đầu khối lệnh hoặc dùng trong slicing |
| `# ...` | Chú thích; Python không thực thi phần sau `#` |
| `def f(...):` | Định nghĩa hàm |
| `return value` | Trả kết quả khỏi hàm |
| `if / elif / else` | Rẽ nhánh theo điều kiện |
| `for x in ...` | Lặp qua các phần tử |
| `import ... as ...` | Nạp thư viện và đặt tên ngắn |

Python dùng **thụt lề** để xác định khối lệnh. Các dòng cùng khối phải thụt giống nhau, thường là 4 dấu cách.

Khung chương trình xử lý ảnh thường gặp:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("duong_dan_anh.png")
if image is None:
    raise FileNotFoundError("Không đọc được ảnh")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Kích thước:", gray.shape)
print("Kiểu dữ liệu:", gray.dtype)

plt.imshow(gray, cmap="gray")
plt.axis("off")
plt.show()
```

Các lệnh kiểm tra môi trường:

```powershell
python --version
python -c "import cv2, numpy, matplotlib; print('Cài đặt thành công')"
python -m pip list
```
