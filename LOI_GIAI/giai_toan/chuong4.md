# Chương 4 — Feature Detection

## I. Bài tập tính toán

### Bài 1 — Phân biệt góc và cạnh bằng ma trận cấu trúc

#### Vì sao trị riêng nhận ra góc?

Trong cửa sổ `W`, lập

```text
H = [ tổng(Ix^2)   tổng(Ix*Iy)
      tổng(Ix*Iy)  tổng(Iy^2) ]
```

`H` mô tả mức thay đổi cường độ khi dịch cửa sổ theo mọi hướng. Hai trị riêng là mức thay đổi theo hai phương chính:

- cả hai nhỏ: vùng phẳng;
- một lớn, một nhỏ: cạnh — đổi mạnh chỉ theo phương vuông góc cạnh;
- cả hai lớn: góc — dịch theo hướng nào cũng thay đổi mạnh.

Đạo hàm dùng

```text
Kx = (1/3) * [ 1  0  -1       Ky = (1/3) * [ 1  1  1
                 1  0  -1                      0  0  0
                 1  0  -1 ]                   -1 -1 -1 ]
```

**Điểm 1:** trong cửa sổ 3×3, slide tính được

```text
H1 = [ 158950   57800
        57800  158950 ]
```

Với ma trận đối xứng dạng `[[a,b],[b,a]]`, trị riêng là `a-b` và `a+b`:

```text
lambda1 = 101150;  lambda2 = 216750
```

Cả hai lớn ⇒ điểm 1 là **góc**.

**Điểm 2:**

```text
H2 = [ 0       0
       0  390150 ]
lambda1 = 0;  lambda2 = 390150
```

Một nhỏ, một lớn ⇒ điểm 2 nằm trên **cạnh**.

---

### Bài 2 — Histogram of Oriented Gradients (HOG)

Độ lớn:

```text
M = [ 20  30  60  90
      45  60  75  50 ]
```

hướng:

```text
Theta (độ) = [ 135   67.5  150  330
                210  240   300   22.5 ]
```

Các bin: `0,45,90,135,180,225,270,315°`.

#### Mấu chốt nội suy phiếu bầu

Nếu `θi < θ < θi+1`, một pixel độ lớn `a` không bỏ toàn bộ vào một bin mà chia tuyến tính:

```text
Mức cộng vào bin i   = ((theta(i+1) - theta) / 45) * a
Mức cộng vào bin i+1 = ((theta - theta(i)) / 45) * a
```

Ví dụ `(a,θ)=(30,67.5°)` nằm đúng giữa `45°` và `90°`, nên mỗi bin nhận 15. Góc là đại lượng vòng tròn: `330°` nằm giữa `315°` và `360°≡0°`, nên chia 60 cho bin 315° và 30 cho bin 0°.

Cộng tám pixel:

```text
h = [55, 40, 15, 60, 35, 70, 45, 110]
```

Chuẩn hóa L2 để giảm ảnh hưởng của độ sáng/độ tương phản chung:

```text
h_normalized = h / sqrt(tổng(h[i]^2))
             ≈ [0.32, 0.24, 0.09, 0.35, 0.21, 0.41, 0.27, 0.65]
```

---

### Bài 3 — Chuẩn hóa vector và SSD

```text
A = [1, 2, 3, 4, 5]
B = [2, 3, 6, 9, 10]
C = [2.5, 6, 7, 9, 12]
```

Nếu so trực tiếp, vector có độ lớn lớn hơn dễ tạo khoảng cách lớn dù “hình dạng” tương tự. Vì vậy chuẩn hóa L2:

```text
A_normalized = A / norm(A)
norm(A) = sqrt(tổng(A[i]^2))
```

Kết quả:

- `Ã ≈ [0.13,0.27,0.40,0.54,0.67]`, `||A||≈7.42`;
- `B̃ ≈ [0.13,0.20,0.40,0.59,0.66]`, `||B||≈15.17`;
- `C̃ ≈ [0.14,0.34,0.39,0.51,0.67]`, `||C||≈17.78`.

Khoảng cách tổng bình phương:

```text
SSD(P, Q) = tổng theo i của (P[i] - Q[i])^2
```

Theo số đã làm tròn trong slide:

```text
SSD(A, B) = 0.0084
SSD(A, C) = 0.0058
SSD(B, C) = 0.027
```

Nhỏ nhất là `SSD(A,C)`, nên **A và C giống nhau nhất**. Nếu dùng toàn bộ chữ số thực, số lẻ hơi khác nhưng thứ tự không đổi.

---

### Bài 4 — Difference of Gaussians và keypoint SIFT

Với các ảnh làm mờ `L_s=G(σ_s)*I`, lập các ảnh sai khác kề nhau:

```text
D[s] = L[s + 1] - L[s]
```

Ví dụ ô trên-trái giữa `s=-1` và `s=0`:

```text
D[-1](0, 0) = 24.15 - 19.15 = 5.00
```

Lặp lại cho mọi ô tạo 5 ma trận DoG. Một ứng viên là keypoint nếu lớn hơn **hoặc nhỏ hơn** toàn bộ 26 hàng xóm: 8 cùng tầng, 9 tầng trên và 9 tầng dưới. Phải tìm cả cực đại và cực tiểu vì blob sáng và blob tối đều quan trọng.

Theo đáp án minh họa của slide, giá trị `-12.44` ở hàng 2, cột 2 của ma trận `D_{-1}=L_0-L_{-1}` được đánh dấu là keypoint vì là cực tiểu cục bộ nổi bật.

Lưu ý học thuật quan trọng: SIFT nghiêm ngặt chỉ kiểm tra tầng DoG có đủ một tầng trên và một tầng dưới. `D_{-1}` đang ở biên của bộ dữ liệu đã cho nên chưa đủ 26 hàng xóm theo chiều scale; muốn xác nhận nghiêm ngặt cần thêm một ảnh DoG trước nó. Với các tầng nội bộ hiện có và điều kiện so sánh nghiêm ngặt, không có cực trị 26-lân-cận. Đây là điểm chưa nhất quán trong slide; khi làm theo đáp án môn học dùng vị trí `-12.44`, khi cài SIFT chuẩn phải áp dụng điều kiện ba tầng.

## II. Bài tập lập trình

Mã hoàn chỉnh: `code/chuong4.py`.

### Bài 1 — Harris corner detector

Harris tránh phải tính trị riêng tại mọi pixel bằng đáp ứng:

```text
R = det(H) - k * trace(H)^2
  = (Sxx*Syy - Sxy^2) - k*(Sxx + Syy)^2
```

Quy trình trong mã:

1. Sobel cho `Ix,Iy`.
2. Lập `Ix²,Iy²,IxIy`.
3. Cộng trong cửa sổ 5×5 để có `Sxx,Syy,Sxy`.
4. Tính `R`, bỏ đáp ứng âm/thấp.
5. NMS 5×5: `R` phải bằng cực đại trong vùng.
6. Sắp giảm dần và lấy tối đa 250 điểm.

`k≈0.04–0.06`: `k` lớn phạt cạnh nhiều hơn nhưng cũng có thể làm mất góc yếu. Mã so sánh với `cv2.cornerHarris` trên `C4_B1.png` và chạy thêm `C4_B1_a.png`.

### Bài 2 — HOG theo cell 4×4

Với từng cell:

1. Tính Sobel và đổi góc về `[0,360)`.
2. Mỗi pixel bỏ phiếu theo magnitude vào hai bin hướng kề nhau.
3. Chuẩn hóa histogram L2.
4. Vẽ đoạn thẳng qua tâm cell; hướng đoạn thẳng là bin, độ dài/tông màu biểu diễn độ lớn.

Nếu cộng 90° vào hướng, mọi vector biểu diễn cũng quay 90°. Điều này minh họa HOG mô tả **hướng cạnh/gradient**, không phải chỉ vị trí vật thể.

## III. Code và lệnh quan trọng của chương 4

### 1. Từ đạo hàm tới ma trận cấu trúc Harris

```python
source = gray.astype(np.float64) / 255.0
ix = cv2.Sobel(source, cv2.CV_64F, 1, 0, ksize=3)
iy = cv2.Sobel(source, cv2.CV_64F, 0, 1, ksize=3)

ix2 = ix * ix
iy2 = iy * iy
ixiy = ix * iy

# Cộng các giá trị trong cửa sổ 5x5
sxx = cv2.boxFilter(ix2,  -1, (5, 5), normalize=False)
syy = cv2.boxFilter(iy2,  -1, (5, 5), normalize=False)
sxy = cv2.boxFilter(ixiy, -1, (5, 5), normalize=False)
```

`normalize=False` nghĩa là lấy tổng trong cửa sổ, không chia cho 25.

### 2. Tính đáp ứng Harris

```python
k = 0.04
det_h = sxx * syy - sxy * sxy
trace_h = sxx + syy
response = det_h - k * trace_h ** 2
```

`** 2` là bình phương. Toàn bộ phép tính trên là theo từng pixel, không phải phép nhân ma trận thông thường.

### 3. NMS và chọn tối đa 250 góc

```python
local_max = cv2.dilate(response, np.ones((5, 5), np.uint8))
mask = (response == local_max) & (response > 0.01 * response.max())

points = np.argwhere(mask)                  # mỗi dòng là [y, x]
scores = response[mask]
order = np.argsort(scores)[::-1]            # giảm dần
points = points[order[:250]]

for y, x in points:
    cv2.circle(result, (int(x), int(y)), 3, (0, 255, 0), 1)
```

`[::-1]` đảo thứ tự; `[:250]` lấy tối đa 250 phần tử đầu. `cv2.circle` nhận tọa độ `(x,y)`, ngược thứ tự truy cập ảnh `[y,x]`.

### 4. Harris có sẵn trong OpenCV

```python
response_cv = cv2.cornerHarris(
    np.float32(gray) / 255.0,
    blockSize=5,
    ksize=3,
    k=0.04,
)
```

### 5. Khung tạo histogram HOG cho mỗi cell

```python
cell_size = 4
bins = np.zeros(8, dtype=float)              # 0,45,...,315 độ

position = (angle % 360) / 45.0
lower = int(np.floor(position)) % 8
upper = (lower + 1) % 8
fraction = position - np.floor(position)

bins[lower] += (1 - fraction) * magnitude
bins[upper] += fraction * magnitude

norm = np.linalg.norm(bins)
if norm > 0:
    bins = bins / norm
```

Dấu `%` là phép chia lấy dư; ở đây nó làm cho bin sau 315° quay lại bin 0°.

### 6. Các hàm hữu ích khi biểu diễn HOG

```python
theta = np.deg2rad(bin_index * 45)
dx = length * np.cos(theta)
dy = length * np.sin(theta)
plt.plot([cx - dx, cx + dx], [cy - dy, cy + dy])
```

### 7. Lệnh chạy

```powershell
cd LOI_GIAI
python .\code\chuong4.py
python .\code\chuong4.py --show
```
