# Bộ lời giải bài tập Xử lý ảnh — chương 2 đến chương 6

Thư mục này gồm lời giải chi tiết và mã Python chạy được cho toàn bộ bài tập trong các PDF:

- [Chương 2 — Image Basics and Filtering](chuong2.md)
- [Chương 3 — Edge Detection](chuong3.md)
- [Chương 4 — Feature Detection](chuong4.md)
- [Chương 5 — Segmentation](chuong5.md)
- [Chương 6 — Deep Learning/MNIST](chuong6.md)

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
