# Chương 2 — Image Basics and Filtering

#### Quick note từ Tuấn: Yoyoyo, hello anh em, hi hi. Tuấn viết bài giải này là lần đầu tiên, và rất có thể dell có lần sau :> Nếu anh em ko cần thì Tuấn giữ những lời giải sau làm của riêng nhé :>>

## I. Bài tập tính toán

### Bài 1 — Đổi ảnh RGB sang ảnh xám

#### Mấu chốt chọn công thức

Mỗi pixel có ba mức màu `(R,G,B)` nhưng ảnh xám chỉ có một mức sáng `I`.

(Màu xám là trường hợp đặc biệt nhé, là chỉ có 1 giá trị thôi, còn pha các màu khác thì vẫn phải điều chỉnh R G B như bình thường)

Ví dụ: màu tím: (R G B) = (255, 0, 255) <--- Pha như bình thường nhé.

Và Fun fact: Mắt người nhạy với xanh lá hơn xanh dương, vì vậy không lấy trung bình đều mà dùng trọng số:

```text
I = 0.299*R + 0.587*G + 0.114*B
```

(Thừa fact: Ba hệ số có tổng bằng 1, nên nếu `R=G=B=k` thì kết quả vẫn bằng `k`.)

Ví dụ pixel đầu `(169,207,11)`:

```text
I = 0.299*169 + 0.587*207 + 0.114*11 = 173.294 ≈ 173
```

Lặp lại độc lập cho 9 pixel, rồi làm tròn về số nguyên:

```text
I_gray = [ 173  162  171
           153  140   66
            65  138  127 ]
```

Lưu ý khi lập trình: đề cho thứ tự RGB; ảnh đọc bởi `cv2.imread` lại có thứ tự BGR.

---

### Bài 2 — Phóng ảnh 2×2 thành 4×4

Ảnh gốc:

```text
I = [ 20  150
      90  200 ]
```

#### Vì sao phải ánh xạ ngược?

Nếu đẩy từng pixel cũ sang ảnh mới, nhiều pixel mới không được gán. (Cái này kết hợp xem ví dụ của thầy nhá). Ta duyệt từng pixel đích `(x_2,y_2)` và truy ngược tọa độ nguồn `(x_1,y_1)`:

```text
x1 = ((W1 - 1) / (W2 - 1)) * x2 = x2 / 3
y1 = ((H1 - 1) / (H2 - 1)) * y2 = y2 / 3
```

(Note: W là width, là số cột của ảnh, H là Height, là số hàng của ảnh) (Số 1 là để chỉ ảnh gốc, Số 2 là để chỉ ảnh đích)
Ví dụ: W1 = 2, H1 = 2, W2 = 4, H2 = 4.
Đối với ảnh gốc là 2x2 và ảnh đích là 4x4

Thế còn tọa độ (x,y) lấy như thế nào? Đơn giản nó là 0 1 2 3 thôi :>
```text
[0,0 1,0 2,0 3,0
 0,1 1,1 2,1 3,1
 0,2 1,2 2,2 3,2
 0,3 1,3 2,3 3,3]
```
Trên là ví dụ về giá trị tọa độ ảnh đích :> 2x2 áp dụng tương tự

Vì chúng ta đang dùng ánh xạ ngược (tìm từ ảnh mới về ảnh gốc), ta cần một "tỷ lệ co giãn" để biết các điểm ảnh nằm tương ứng ở đâu. Phân số `(W1 - 1) / (W2 - 1)` chính là tỷ lệ đó. Nó lấy "tọa độ tối đa của ảnh gốc" chia cho "tọa độ tối đa của ảnh mới".

Việc dùng công thức `W - 1`, `H - 1` làm hai đầu biên khớp chính xác: `x_2=3` ánh xạ đúng tới `x_1=1`, không vượt ảnh.

#### a. Nội suy gần nhất

Làm tròn `(x_1,y_1)` tới pixel nguồn gần nhất:

```text
I_NN = [ 20  20  150  150
         20  20  150  150
         90  90  200  200
         90  90  200  200 ]
```

Mấu chốt: phương pháp này chỉ sao chép giá trị nên nhanh nhưng tạo răng cưa. Nếu hỏi Tuấn răng cưa trông thế nào thì chơi game vào setting, tắt sửa răng cưa đi là biết :> Nó trông ngứa mắt lắm.

#### b. Nội suy song tuyến

Nhớ công thức trên ko? Ví dụ nếu: x2 = 1 thì x1 = 1/3 đúng ko? Là 0.3333. NẾU ta dùng "Nội suy gần nhất" thì nó sẽ nhận pixel tọa độ có x = 0 thôi, còn phần thập phân bị làm tròn rồi.

Vậy nên phải làm thế nào?

"Nội suy song tuyến" giải quyết điểm yếu đó. Thay vì chỉ lấy màu của 1 điểm duy nhất, nó sẽ lấy tọa độ lẻ đó làm trung tâm và pha trộn màu của cả 4 điểm ảnh (pixel) bao quanh nó.

Với điểm nguồn `(x,y)` nằm giữa bốn pixel `Q11, Q21, Q12, Q22`, đặt:

```text
dx = x - floor(x)
dy = y - floor(y)
```
floor(x) là làm tròn x xuống, vd: 0.667 làm tròn thành 0

Khi đó

```text
f(x, y) = (1-dx)*(1-dy)*Q11 + dx*(1-dy)*Q21
        + (1-dx)*dy*Q12     + dx*dy*Q22
```
(Đậu xanh công thức khó nhớ thật :>)
Vậy Q11, Q21, Q12, Q22 lấy ở đâu? Ở ảnh đầu 2x2 đó :> Q11 = 20, Q21 = 150, Q12 = 90, Q22 = 200

Ví dụ tại tọa độ nguồn `(0.33,0.33)`:

```text
f ≈ 0.67^2*20 + 0.33*0.67*150
  + 0.67*0.33*90 + 0.33^2*200
  = 83.822 ≈ 84
```

Theo cách làm tròn `0.33/0.67` của slide:

```text
I_bilinear = [ 20   63  107  150
                43   84  126  167
                67  106  144  183
                90  127  163  200 ]
```

Giữ chính xác `1/3,2/3` tới cuối có thể cho vài ô lệch 1 đơn vị; đó chỉ là khác biệt làm tròn.

---

### Bài 3 — Quay các điểm 30°

Đề ghi tọa độ `(hàng,cột)`, còn công thức dùng `(x=cột,y=hàng)`. Đây là mấu chốt quan trọng nhất.

Với quy ước vector hàng trong đề:

```text
[x  y  1] = [v  w  1] * [ cos(theta)   sin(theta)  0
                        -sin(theta)    cos(theta)  0
                         0             0           1 ]
```

suy ra

```text
x = v*cos(theta) - w*sin(theta)
y = v*sin(theta) + w*cos(theta)
```

Vì `cos30°≈0.866`, `sin30°=0.5`:

- `A(hàng=10,cột=10)` → dùng `(v,w)=(10,10)` → `(x,y)=(3.66,13.66)` → trả về `(hàng,cột)=(14,4)`.
- `B(15,10)` → `(v,w)=(10,15)` → `(1.16,17.99)` → `(18,1)`.
- `C(10,15)` → `(v,w)=(15,10)` → `(7.99,16.16)` → `(16,8)`.

Phép quay trên quay quanh gốc `(0,0)`. Muốn quay quanh tâm `(c_x,c_y)`, phải tịnh tiến về gốc, quay, rồi tịnh tiến trở lại.

---

### Bài 4 — Tích chập và lọc trung vị

Ảnh:

```text
I = [  13  131   22
      255  159   61
      197   79  176 ]
```

#### a. Tích chập

Với kernel 3×3, thêm một viền 0. Tích chập đúng nghĩa phải lật kernel theo cả hai chiều, sau đó lấy tổng tích từng phần tử.

Ví dụ ô trên-trái với `s_x` sau khi lật:

```text
(-1)*0 + 0*0 + 1*0 + (-2)*0 + 0*13 + 2*131 + (-1)*0 + 0*255 + 1*159 = 421
```

Kết quả:

```text
I * s_x = [ 421  -176  -421
            528  -400  -528
            317  -236  -317 ]
```

```text
I * s_y = [ 669   634   281
            316   234   256
           -669  -634  -281 ]
```

```text
I * G = [  68.327   73.085   40.638
          126.351  134.329   77.361
          105.031  112.501   70.052 ]
```

Nói chung là xem ví dụ thầy nó dễ hiểu hơn, Tuấn ngu quá.

#### b. Lọc trung vị 3×3

Tiếp tục là tạo 1 viền 0:

Tại mỗi vị trí, sắp xếp 9 giá trị trong cửa sổ và lấy giá trị thứ 5:

```text
I_median = [  0   22   0
             79  131  61
              0   79   0 ]
```

## II. Bài tập lập trình

Mã hoàn chỉnh nằm trong `../code/chuong2.py`.

### Bài 1 — Đọc, hiển thị kênh màu, đổi xám

- OpenCV trả về `shape=(H,W,3)` và thứ tự BGR.
- Matplotlib cần RGB, vì vậy dùng `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
- Đổi xám thủ công phải tách đúng `B,G,R` rồi tính `0.299R+0.587G+0.114B`.
- So sánh bằng sai số tuyệt đối. Sai số 0–1 là bình thường do OpenCV dùng số học nguyên/làm tròn tối ưu.

### Bài 2 — Phóng to thủ công

Hai hàm `resize_nearest` và `resize_bilinear` đều dùng ánh xạ ngược. Với bilinear, cần chặn `x1,y1` ở biên và tính trên `float` trước khi `clip` về `[0,255]`.

### Bài 3 — Histogram

Histogram `h[k]` là số pixel có mức xám `k`:

```text
h(k) = số lượng vị trí (x, y) mà I(x, y) = k, với k từ 0 đến 255.
```

Trong mã, `np.bincount(gray.ravel(), minlength=256)` chính là công thức trên được vector hóa.

### Bài 4 — Quay ảnh 30°

Dùng ánh xạ ngược từ pixel đích về nguồn để không sinh lỗ. Trước/sau phép quay phải trừ/cộng tâm ảnh. Mã có cả nội suy song tuyến thủ công và `cv2.warpAffine` để đối chiếu.

### Bài 5 — Ba bộ lọc 7×7

- Trung bình: mọi phần tử bằng `1/49`; làm trơn đều nhưng nhòe cạnh.
- Gaussian `σ=10`: trọng số giảm theo khoảng cách tâm; với kernel chỉ 7×7 và `σ` rất lớn, trọng số khá gần bộ lọc trung bình.
- Trung vị: lấy median; giữ cạnh tốt hơn và mạnh với nhiễu xung.

Để so sánh công bằng, dùng cùng kích thước 7×7 và cùng ảnh đầu vào; quan sát vùng biên, chi tiết nhỏ và nhiễu thay vì chỉ nhìn toàn ảnh.

## III. Code, ký hiệu và lệnh quan trọng của chương 2

### 1. Đọc ảnh và kiểm tra dữ liệu

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C2_B1.png")
if img is None:
    raise FileNotFoundError("Không đọc được C2_B1.png")

height, width, channels = img.shape
print("Cao, rộng, số kênh:", height, width, channels)
print("Kiểu dữ liệu:", img.dtype)
print("Pixel đầu tiên [B,G,R]:", img[0, 0])
```

Ký hiệu cần nhớ:

- `img.shape` trả `(số hàng, số cột, số kênh)`.
- `img[y, x]` truy cập pixel hàng `y`, cột `x`.
- `img[:, :, 0]` lấy toàn bộ kênh B; `1` là G; `2` là R.
- `:` nghĩa là lấy toàn bộ trên chiều đó.
- `dtype` thường là `uint8`, tức số nguyên từ 0 đến 255.

### 2. Hiển thị đúng màu bằng Matplotlib

```python
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(rgb)
plt.title("Ảnh RGB")
plt.axis("off")
plt.show()
```

OpenCV đọc BGR nhưng Matplotlib hiểu RGB, vì vậy phải đổi thứ tự kênh trước khi hiển thị.

### 3. Hiển thị riêng từng kênh màu

```python
b, g, r = cv2.split(img)
zeros = np.zeros_like(b)

red_image = cv2.merge([zeros, zeros, r])
green_image = cv2.merge([zeros, g, zeros])
blue_image = cv2.merge([b, zeros, zeros])
```

`split` tách các kênh; `merge` ghép ba kênh lại. Thứ tự của `merge` vẫn là BGR.

### 4. Đổi ảnh màu sang xám theo hai cách

```python
# Cách 1: hàm có sẵn
gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Cách 2: tự áp dụng công thức
b_float = b.astype(np.float64)
g_float = g.astype(np.float64)
r_float = r.astype(np.float64)

gray_float = 0.299 * r_float + 0.587 * g_float + 0.114 * b_float
gray_manual = np.clip(np.rint(gray_float), 0, 255).astype(np.uint8)

difference = cv2.absdiff(gray_cv, gray_manual)
print("Sai số lớn nhất:", difference.max())
print("Sai số trung bình:", difference.mean())
```

- `astype(np.float64)` tránh tràn số khi tính.
- `np.rint` làm tròn về số nguyên gần nhất.
- `np.clip(...,0,255)` chặn giá trị hợp lệ của pixel.
- `cv2.absdiff` tính chênh lệch tuyệt đối giữa hai ảnh.

### 5. Ánh xạ ngược khi thay đổi kích thước

```python
def source_coordinate(dst_index, src_size, dst_size):
    if dst_size <= 1:
        return 0.0
    return dst_index * (src_size - 1) / (dst_size - 1)
```

Hàm này biến vị trí trên ảnh đích thành vị trí tương ứng trên ảnh nguồn. Dùng `size-1` để hai đầu biên khớp chính xác.

### 6. Nội suy gần nhất

```python
old_h, old_w = img.shape[:2]
new_h, new_w = 2 * old_h, 2 * old_w
nearest = np.zeros((new_h, new_w, 3), dtype=img.dtype)

for y2 in range(new_h):
    y1 = round(source_coordinate(y2, old_h, new_h))
    for x2 in range(new_w):
        x1 = round(source_coordinate(x2, old_w, new_w))
        nearest[y2, x2] = img[y1, x1]
```

`range(new_h)` tạo các chỉ số từ 0 đến `new_h-1`. `round` chọn pixel nguồn gần nhất.

### 7. Nội suy song tuyến

```python
def bilinear_sample(image, x, y):
    h, w = image.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0

    return (
        (1 - dx) * (1 - dy) * image[y0, x0]
        + dx * (1 - dy) * image[y0, x1]
        + (1 - dx) * dy * image[y1, x0]
        + dx * dy * image[y1, x1]
    )
```

Bốn dòng nhân là bốn trọng số của bốn pixel xung quanh. Tổng bốn trọng số luôn bằng 1.

### 8. So sánh với `cv2.resize`

```python
nearest_cv = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
bilinear_cv = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
```

OpenCV nhận kích thước theo `(width,height)`, trong khi `shape` trả `(height,width,channels)`.

### 9. Tính histogram ảnh xám

```python
histogram = np.bincount(gray_cv.ravel(), minlength=256)

plt.plot(np.arange(256), histogram)
plt.xlim(0, 255)
plt.xlabel("Mức xám")
plt.ylabel("Số pixel")
plt.grid()
plt.show()
```

- `ravel()` dàn ma trận ảnh thành một vector.
- `bincount` đếm số lần xuất hiện của từng giá trị.
- `minlength=256` đảm bảo đủ các mức xám 0–255.

### 10. Quay ảnh 30 độ bằng OpenCV

```python
h, w = img.shape[:2]
center = ((w - 1) / 2, (h - 1) / 2)

rotation_matrix = cv2.getRotationMatrix2D(
    center=center,
    angle=30,
    scale=1.0,
)

rotated = cv2.warpAffine(
    img,
    rotation_matrix,
    (w, h),
    flags=cv2.INTER_LINEAR,
)
```

`warpAffine` dùng ma trận biến đổi 2×3. Kích thước đầu ra vẫn truyền theo `(width,height)`.

### 11. Tích chập và lưu ý phải lật kernel

```python
kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1],
], dtype=np.float64)

# filter2D tính tương quan, nên lật kernel để thành tích chập đúng nghĩa
kernel_flipped = np.flip(kernel, axis=(0, 1))
filtered = cv2.filter2D(
    gray_cv.astype(np.float64),
    cv2.CV_64F,
    kernel_flipped,
    borderType=cv2.BORDER_CONSTANT,
)
```

`axis=(0,1)` nghĩa là lật theo cả chiều hàng và chiều cột.

### 12. Ba bộ lọc kích thước 7×7

```python
mean_img = cv2.blur(img, (7, 7))
gaussian_img = cv2.GaussianBlur(img, (7, 7), sigmaX=10, sigmaY=10)
median_img = cv2.medianBlur(img, 7)
```

- `(7,7)` là kích thước kernel của trung bình và Gaussian.
- Median nhận một số nguyên `7`, không nhận tuple.
- Kích thước kernel phải là số lẻ để có pixel tâm.

### 13. Lưu kết quả

```python
success = cv2.imwrite("ket_qua.png", rotated)
print("Lưu thành công:", success)
```

### 14. Lệnh chạy chương 2

Đứng tại thư mục `LOI_GIAI`:

```powershell
python .\code\chuong2.py
python .\code\chuong2.py --show
```

Kiểm tra nhanh thư viện:

```powershell
python -c "import cv2, numpy, matplotlib; print('OK')"
```
