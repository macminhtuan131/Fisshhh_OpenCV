# Chương 2 — Image Basics and Filtering

## I. Bài tập tính toán

### Bài 1 — Đổi ảnh RGB sang ảnh xám

#### Mấu chốt chọn công thức

Mỗi pixel có ba mức màu `(R,G,B)` nhưng ảnh xám chỉ có một mức sáng `I`. Mắt người nhạy với xanh lá hơn xanh dương, vì vậy không lấy trung bình đều mà dùng trọng số:

\[
I=0.299R+0.587G+0.114B.
\]

Ba hệ số có tổng bằng 1, nên nếu `R=G=B=k` thì kết quả vẫn bằng `k`.

Ví dụ pixel đầu `(169,207,11)`:

\[
I=0.299(169)+0.587(207)+0.114(11)=173.294\approx173.
\]

Lặp lại độc lập cho 9 pixel, rồi làm tròn về số nguyên:

\[
I_{gray}=\begin{bmatrix}
173&162&171\\
153&140&66\\
65&138&127
\end{bmatrix}.
\]

Điểm dễ sai: đề cho thứ tự RGB; ảnh đọc bởi `cv2.imread` lại có thứ tự BGR.

---

### Bài 2 — Phóng ảnh 2×2 thành 4×4

Ảnh gốc:

\[
I=\begin{bmatrix}20&150\\90&200\end{bmatrix}.
\]

#### Vì sao phải ánh xạ ngược?

Nếu đẩy từng pixel cũ sang ảnh mới, nhiều pixel mới không được gán. Ta duyệt từng pixel đích `(x_2,y_2)` và truy ngược tọa độ nguồn `(x_1,y_1)`:

\[
x_1=\frac{W_1-1}{W_2-1}x_2=\frac{x_2}{3},\qquad
y_1=\frac{H_1-1}{H_2-1}y_2=\frac{y_2}{3}.
\]

Việc dùng `W-1`, `H-1` làm hai đầu biên khớp chính xác: `x_2=3` ánh xạ đúng tới `x_1=1`, không vượt ảnh.

#### a. Nội suy gần nhất

Làm tròn `(x_1,y_1)` tới pixel nguồn gần nhất:

\[
I_{NN}=\begin{bmatrix}
20&20&150&150\\
20&20&150&150\\
90&90&200&200\\
90&90&200&200
\end{bmatrix}.
\]

Mấu chốt: phương pháp này chỉ sao chép giá trị nên nhanh nhưng tạo khối/răng cưa.

#### b. Nội suy song tuyến

Với điểm nguồn `(x,y)` nằm giữa bốn pixel `Q11, Q21, Q12, Q22`, đặt

\[
d_x=x-\lfloor x\rfloor,\qquad d_y=y-\lfloor y\rfloor.
\]

Khi đó

\[
f(x,y)=(1-d_x)(1-d_y)Q_{11}+d_x(1-d_y)Q_{21}
 +(1-d_x)d_yQ_{12}+d_xd_yQ_{22}.
\]

Các hệ số chính là mức “gần” từng góc và luôn có tổng bằng 1. Ví dụ tại tọa độ nguồn `(0.33,0.33)`:

\[
f\approx0.67^2(20)+0.33(0.67)(150)+0.67(0.33)(90)+0.33^2(200)
=83.822\approx84.
\]

Theo cách làm tròn `0.33/0.67` của slide:

\[
I_{bilinear}=\begin{bmatrix}
20&63&107&150\\
43&84&126&167\\
67&106&144&183\\
90&127&163&200
\end{bmatrix}.
\]

Giữ chính xác `1/3,2/3` tới cuối có thể cho vài ô lệch 1 đơn vị; đó chỉ là khác biệt làm tròn.

---

### Bài 3 — Quay các điểm 30°

Đề ghi tọa độ `(hàng,cột)`, còn công thức dùng `(x=cột,y=hàng)`. Đây là mấu chốt quan trọng nhất.

Với quy ước vector hàng trong đề:

\[
[x\ y\ 1]=[v\ w\ 1]
\begin{bmatrix}
\cos\theta&\sin\theta&0\\
-\sin\theta&\cos\theta&0\\
0&0&1
\end{bmatrix},
\]

suy ra

\[
x=v\cos\theta-w\sin\theta,\qquad
y=v\sin\theta+w\cos\theta.
\]

Vì `cos30°≈0.866`, `sin30°=0.5`:

- `A(hàng=10,cột=10)` → dùng `(v,w)=(10,10)` → `(x,y)=(3.66,13.66)` → trả về `(hàng,cột)=(14,4)`.
- `B(15,10)` → `(v,w)=(10,15)` → `(1.16,17.99)` → `(18,1)`.
- `C(10,15)` → `(v,w)=(15,10)` → `(7.99,16.16)` → `(16,8)`.

Phép quay trên quay quanh gốc `(0,0)`. Muốn quay quanh tâm `(c_x,c_y)`, phải tịnh tiến về gốc, quay, rồi tịnh tiến trở lại.

---

### Bài 4 — Tích chập và lọc trung vị

Ảnh:

\[
I=\begin{bmatrix}13&131&22\\255&159&61\\197&79&176\end{bmatrix}.
\]

#### a. Tích chập

Với kernel 3×3, thêm một viền 0. Tích chập đúng nghĩa phải lật kernel theo cả hai chiều, sau đó lấy tổng tích từng phần tử.

Ví dụ ô trên-trái với `s_x` sau khi lật:

\[
(-1)0+0(0)+1(0)+(-2)0+0(13)+2(131)+(-1)0+0(255)+1(159)=421.
\]

Kết quả:

\[
I*s_x=\begin{bmatrix}
421&-176&-421\\
528&-400&-528\\
317&-236&-317
\end{bmatrix},
\]

\[
I*s_y=\begin{bmatrix}
669&634&281\\
316&234&256\\
-669&-634&-281
\end{bmatrix},
\]

\[
I*G=\begin{bmatrix}
68.327&73.085&40.638\\
126.351&134.329&77.361\\
105.031&112.501&70.052
\end{bmatrix}.
\]

`s_x,s_y` có hệ số dương/âm nên đo thay đổi theo hướng; `G` có hệ số dương và tổng gần 1 nên làm trơn.

#### b. Lọc trung vị 3×3

Tại mỗi vị trí, sắp xếp 9 giá trị trong cửa sổ và lấy giá trị thứ 5. Với quy ước đệm 0 của đề:

\[
I_{median}=\begin{bmatrix}
0&22&0\\
79&131&61\\
0&79&0
\end{bmatrix}.
\]

Lọc trung vị phi tuyến, không dùng phép nhân kernel. Nó đặc biệt tốt với nhiễu muối-tiêu vì giá trị ngoại lai bị loại khi lấy trung vị.

## II. Bài tập lập trình

Mã hoàn chỉnh nằm trong `code/chuong2.py`.

### Bài 1 — Đọc, hiển thị kênh màu, đổi xám

- OpenCV trả về `shape=(H,W,3)` và thứ tự BGR.
- Matplotlib cần RGB, vì vậy dùng `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
- Đổi xám thủ công phải tách đúng `B,G,R` rồi tính `0.299R+0.587G+0.114B`.
- So sánh bằng sai số tuyệt đối. Sai số 0–1 là bình thường do OpenCV dùng số học nguyên/làm tròn tối ưu.

### Bài 2 — Phóng to thủ công

Hai hàm `resize_nearest` và `resize_bilinear` đều dùng ánh xạ ngược. Với bilinear, cần chặn `x1,y1` ở biên và tính trên `float` trước khi `clip` về `[0,255]`.

### Bài 3 — Histogram

Histogram `h[k]` là số pixel có mức xám `k`:

\[
h(k)=\#\{(x,y):I(x,y)=k\},\quad k=0,\ldots,255.
\]

Trong mã, `np.bincount(gray.ravel(), minlength=256)` chính là công thức trên được vector hóa.

### Bài 4 — Quay ảnh 30°

Dùng ánh xạ ngược từ pixel đích về nguồn để không sinh lỗ. Trước/sau phép quay phải trừ/cộng tâm ảnh. Mã có cả nội suy song tuyến thủ công và `cv2.warpAffine` để đối chiếu.

### Bài 5 — Ba bộ lọc 7×7

- Trung bình: mọi phần tử bằng `1/49`; làm trơn đều nhưng nhòe cạnh.
- Gaussian `σ=10`: trọng số giảm theo khoảng cách tâm; với kernel chỉ 7×7 và `σ` rất lớn, trọng số khá gần bộ lọc trung bình.
- Trung vị: lấy median; giữ cạnh tốt hơn và mạnh với nhiễu xung.

Để so sánh công bằng, dùng cùng kích thước 7×7 và cùng ảnh đầu vào; quan sát vùng biên, chi tiết nhỏ và nhiễu thay vì chỉ nhìn toàn ảnh.
