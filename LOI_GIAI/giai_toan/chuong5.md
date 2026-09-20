# Chương 5 — Segmentation

## I. Bài tập tính toán

### Bài 1 — K-means hai cụm

Các điểm:

`p1=(1,1), p2=(1,2), p3=(2,1), p4=(2,2), p5=(5,5), p6=(5,6), p7=(6,5), p8=(6,6), p9=(4.5,4.5), p10=(0.5,0.5)`.

Tâm đầu: `C1=(1,6)`, `C2=(6,1)`.

#### Mấu chốt của K-means

K-means tối thiểu hóa tổng bình phương khoảng cách từ điểm tới tâm cụm:

```text
J = tổng theo i của [khoảng cách từ p[i] đến tâm cụm của p[i]]^2
```

Vì vậy hai bước phải lặp:

1. Gán mỗi điểm vào tâm gần nhất theo
   ```text
   d(p, c) = sqrt((x_p - x_c)^2 + (y_p - y_c)^2)
   ```
2. Cập nhật mỗi tâm bằng trung bình các điểm trong cụm:
   ```text
   c[j] = (tổng các điểm trong cụm j) / (số điểm của cụm j)
   ```

Nếu khoảng cách hòa, kết quả phụ thuộc quy tắc phá hòa. Slide gán hòa vào `C2` ở lượt đầu.

**Lượt 1 theo slide:**

- `C1`: `p2,p6` → tâm mới `(3,4)`.
- `C2`: các điểm còn lại → tâm mới `(3.375,3.125)`.

**Lượt 2:** sau khi tính lại khoảng cách:

- cụm thấp: `p1,p2,p3,p4,p10` → tâm `(1.3,1.3)`;
- cụm cao: `p5,p6,p7,p8,p9` → tâm `(5.3,5.3)`.

**Lượt 3:** phép gán không đổi, tâm vẫn `(1.3,1.3)` và `(5.3,5.3)` nên dừng.

Kết quả cuối:

```text
Cụm 1: S1 = {p1, p2, p3, p4, p10}; tâm c1 = (1.3, 1.3)
Cụm 2: S2 = {p5, p6, p7, p8, p9};  tâm c2 = (5.3, 5.3)
```

Tên nhãn cụm có thể bị hoán đổi mà nghiệm vẫn tương đương; ý nghĩa nằm ở các nhóm điểm, không nằm ở số 1/2.

---

### Bài 2 — Dilation với hai phần tử cấu trúc

Quy ước pixel đen là 1, trắng là 0:

```text
I = [ 0  0  0  0  0
      0  1  0  0  0
      0  1  1  1  0
      0  0  0  1  0
      0  0  0  0  0 ]
```

Hai phần tử cấu trúc:

```text
F1 = [ 1  1  1       F2 = [ 0  1  0
       1  1  1              1  1  1
       1  1  1 ]            0  1  0 ]
```

#### Vì sao dilation “nở” vật thể?

Tại mỗi tâm, đầu ra bằng 1 nếu có **ít nhất một** vị trí mà ảnh và phần tử cấu trúc cùng bằng 1. Tương đương, mỗi pixel 1 của ảnh đóng dấu hình `F` quanh nó. Đệm ngoài ảnh bằng 0.

Kết quả với khối vuông:

```text
I dilation F1 = [ 1  1  1  0  0
                  1  1  1  1  1
                  1  1  1  1  1
                  1  1  1  1  1
                  0  0  1  1  1 ]
```

Kết quả với hình chữ thập:

```text
I dilation F2 = [ 0  1  0  0  0
                  1  1  1  1  0
                  1  1  1  1  1
                  0  1  1  1  1
                  0  0  0  1  0 ]
```

`F1` nở cả chéo; `F2` chỉ nở bốn hướng chính, cho thấy hình phần tử cấu trúc quyết định hình học kết quả.

## II. Bài tập lập trình

Mã hoàn chỉnh: `code/chuong5.py`.

### Bài 1 — K-means dữ liệu 2D thành ba cụm

Đọc `C5_B1.txt` bằng `np.loadtxt`. Mã cài trực tiếp vòng lặp gán/cập nhật và in inertia mỗi lượt. Điều kiện dừng là dịch chuyển tâm nhỏ hơn `tol`, không nên so sánh số thực bằng đúng tuyệt đối.

Một cụm rỗng có thể xảy ra nếu khởi tạo xấu; mã xử lý bằng cách lấy điểm xa tâm gần nhất làm tâm mới. `seed` cố định giúp tái lập kết quả.

### Bài 2 — Phân ngưỡng từ K-means trên histogram

Mỗi mức xám `g` tạo điểm hai chiều `(g, h[g])`. Hai chiều có đơn vị/range rất khác nên cần chuẩn hóa trước khi tính khoảng cách; nếu không, tần số lớn có thể lấn át độ xám hoặc ngược lại.

Sau khi phân hai cụm, sắp các tâm theo thành phần mức xám. Ngưỡng được chọn ở trung điểm giữa mức xám trung bình của hai cụm:

```text
T = (mu_dark + mu_bright) / 2
```

So với ngưỡng cố định 127 để thấy K-means thích nghi với histogram của chính ảnh. Nếu phân mảnh chưa tốt, có thể thêm đặc trưng vị trí `(x,y)`, lọc nhiễu trước, dùng nhiều cụm hơn, hoặc áp dụng opening/closing sau ngưỡng.

### Bài 3 — K-means màu với K=5

Biến ảnh `(H,W,3)` thành `N×3`, mỗi pixel là một điểm trong không gian BGR/RGB. K-means trả mỗi pixel về một trong năm màu tâm. Đây là lượng tử hóa màu và cũng tạo phân vùng theo màu.

Hạn chế: hai vùng xa nhau nhưng cùng màu bị xem là một cụm. Muốn xét tính liên tục không gian, dùng vector `[R,G,B,λx,λy]`.

### Bài 4 — Threshold và morphology

Với ảnh nhị phân `H` và phần tử cấu trúc `F`:

- Dilation `H⊕F`: chỉ cần một pixel 1 → nở vật thể, lấp khe nhỏ.
- Erosion `H⊖F`: mọi vị trí được đánh dấu bởi `F` phải là 1 → co vật thể, loại chấm nhỏ.
- Opening `(H⊖F)⊕F`: erosion rồi dilation → loại nhiễu sáng nhỏ, tách cầu mảnh.
- Closing `(H⊕F)⊖F`: dilation rồi erosion → lấp lỗ/khe tối nhỏ.

Mã có cả phiên bản tự viết bằng cửa sổ và phép gọi OpenCV để đối chiếu.

## III. Code và lệnh quan trọng của chương 5

### 1. Đọc dữ liệu điểm 2D

```python
import numpy as np

points = np.loadtxt("C5_B1.txt", dtype=np.float64)
print(points.shape)       # (số điểm, 2)
print(points[:5])         # xem 5 điểm đầu
```

`points[:5]` là slicing: lấy từ đầu đến trước vị trí 5.

### 2. Bước gán cụm của K-means

```python
# points: (N,D), centers: (K,D)
difference = points[:, None, :] - centers[None, :, :]
squared_distance = np.sum(difference ** 2, axis=2)
labels = np.argmin(squared_distance, axis=1)
```

Ý nghĩa kích thước:

- `points[:, None, :]` có dạng `(N,1,D)`.
- `centers[None, :, :]` có dạng `(1,K,D)`.
- NumPy broadcasting tạo hiệu của mọi cặp điểm–tâm, dạng `(N,K,D)`.
- `argmin(..., axis=1)` trả chỉ số tâm gần nhất của từng điểm.

### 3. Bước cập nhật tâm

```python
new_centers = np.empty_like(centers)

for cluster in range(k):
    members = points[labels == cluster]
    if len(members) > 0:
        new_centers[cluster] = members.mean(axis=0)
```

`labels == cluster` tạo mặt nạ True/False; dùng mặt nạ trong `points[...]` để lấy các điểm thuộc cụm.

### 4. Điều kiện dừng K-means

```python
movement = np.linalg.norm(new_centers - centers, axis=1).max()
centers = new_centers

if movement < 1e-5:
    break
```

`break` thoát khỏi vòng lặp gần nhất. Không nên dùng `new_centers == centers` vì số thực có sai số.

### 5. Histogram và phân ngưỡng

```python
gray = cv2.imread("C5_B2.PNG", cv2.IMREAD_GRAYSCALE)
histogram = np.bincount(gray.ravel(), minlength=256)

threshold = 150
binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
```

`np.where(điều_kiện, giá_trị_đúng, giá_trị_sai)` thực hiện cho mọi pixel cùng lúc.

### 6. K-means màu

```python
h, w = image.shape[:2]
pixels = image.reshape(-1, 3).astype(np.float64)

labels, centers, history = kmeans(pixels, k=5)
segmented_pixels = centers[labels]
segmented = segmented_pixels.reshape(h, w, 3).astype(np.uint8)
```

`reshape(-1,3)` để Python tự tính số hàng; mỗi hàng mới là một pixel BGR.

### 7. Dilation và erosion tự viết

```python
padded = np.pad(binary > 0, 1, mode="constant", constant_values=False)
result = np.zeros_like(binary)

for y in range(binary.shape[0]):
    for x in range(binary.shape[1]):
        window = padded[y:y + 3, x:x + 3]
        result[y, x] = 255 if np.any(window) else 0       # dilation
        # result[y, x] = 255 if np.all(window) else 0    # erosion
```

`np.any` đúng nếu ít nhất một phần tử đúng; `np.all` chỉ đúng nếu tất cả phần tử đúng.

### 8. Morphology bằng OpenCV

```python
kernel = np.ones((3, 3), dtype=np.uint8)

dilation = cv2.dilate(binary, kernel)
erosion = cv2.erode(binary, kernel)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

### 9. Lệnh chạy

```powershell
cd LOI_GIAI
python .\code\chuong5.py
python .\code\chuong5.py --show
```
