# Chương 3 — Edge Detection

## I. Bài tập tính toán

### Bài 1 — Đạo hàm rời rạc

#### Mấu chốt chọn công thức

Cạnh là nơi cường độ thay đổi nhanh, còn đạo hàm đo chính sự thay đổi. Với dãy mức xám `f[x]`, đề dùng:

\[
D_b(x)=f(x)-f(x-1),\quad
D_f(x)=f(x)-f(x+1),\quad
D_c(x)=f(x+1)-f(x-1).
\]

Lưu ý `D_f` trong slide ngược dấu với cách viết thường gặp `f(x+1)-f(x)`. Dấu cho biết chiều sáng→tối hay tối→sáng; độ lớn tuyệt đối mới cho biết cạnh mạnh hay yếu.

Chỉ tính cho 10 phần tử giữa vì đạo hàm giữa cần cả hai hàng xóm. Kết quả ba hàng:

**Hàng 1** `10,74,228,239,198,126,25,17,62,159,221,220`:

- Lùi: `[64,154,11,-41,-72,-101,-8,45,97,62]`
- Tiến theo đề: `[-154,-11,41,72,101,8,-45,-97,-62,1]`
- Giữa: `[218,165,-30,-113,-173,-109,37,142,159,61]`

**Hàng 2** `92,102,102,88,55,50,87,180,159,177,186,213`:

- Lùi: `[10,0,-14,-33,-5,37,93,-21,18,9]`
- Tiến: `[0,14,33,5,-37,-93,21,-18,-9,-27]`
- Giữa: `[10,-14,-47,-38,32,130,72,-3,27,36]`

**Hàng 3** `124,124,114,101,76,21,26,39,68,149,168,173`:

- Lùi: `[0,-10,-13,-25,-55,5,13,29,81,19]`
- Tiến: `[10,13,25,55,-5,-13,-29,-81,-19,-5]`
- Giữa: `[-10,-23,-38,-80,-50,18,42,110,100,24]`

Các vị trí có `|D|` lớn có khả năng chứa cạnh. Không có ngưỡng trong đề nên không thể gắn nhãn cạnh tuyệt đối; có thể xếp hạng theo `|D|`. Đạo hàm giữa nhìn qua hai pixel nên đáp ứng thường lớn hơn và vị trí cạnh có thể rộng hơn.

---

### Bài 2 — Sobel, độ lớn và hướng gradient

Ảnh:

\[
I=\begin{bmatrix}66&132&42\\80&38&53\\178&138&106\end{bmatrix}.
\]

Sobel kết hợp đạo hàm theo một hướng và làm trơn theo hướng vuông góc. Đề chia kernel cho 8. Với đệm 0 và tích chập, kết quả:

\[
I_x=\frac18\begin{bmatrix}
170&-117&-302\\308&-203&-346\\176&-277&-314
\end{bmatrix}
=\begin{bmatrix}
21.25&-14.63&-37.75\\38.50&-25.38&-43.25\\22.00&-34.63&-39.25
\end{bmatrix},
\]

\[
I_y=\frac18\begin{bmatrix}
198&209&144\\230&188&134\\-198&-209&-144
\end{bmatrix}
=\begin{bmatrix}
24.75&26.13&18.00\\28.75&23.50&16.75\\-24.75&-26.13&-18.00
\end{bmatrix}.
\]

Mỗi pixel có vector gradient `(Ix,Iy)`. Do đó:

\[
A=\sqrt{I_x^2+I_y^2},\qquad \theta=\operatorname{atan2}(I_y,I_x).
\]

Phải dùng `atan2`, không chỉ `atan(Iy/Ix)`, vì `atan2` giữ đúng góc phần tư và xử lý `Ix=0`.

\[
A=\begin{bmatrix}
32.62&29.94&41.82\\48.05&34.59&46.38\\33.11&43.38&43.18
\end{bmatrix},
\]

\[
\theta(^{\circ})=\begin{bmatrix}
49.35&119.24&154.51\\36.75&137.20&158.83\\-48.37&-142.96&-155.36
\end{bmatrix}.
\]

Ví dụ ô đầu: `sqrt(21.25²+24.75²)=32.62`, `atan2(24.75,21.25)=49.35°`.

---

### Bài 3 — Non-Maximum Suppression (NMS)

#### Mấu chốt

Gradient vuông góc với cạnh. Muốn làm cạnh mảnh, tại pixel trung tâm chỉ so sánh độ lớn với hai hàng xóm **dọc theo hướng gradient**:

- `0°`: trái/phải;
- `45°`: chéo dưới-trái/trên-phải;
- `90°`: trên/dưới;
- `135°`: chéo trên-trái/dưới-phải.

Nếu không nhỏ hơn cả hai thì giữ, ngược lại gán 0. Chỉ tính vùng 3×3 tô xám ở giữa. Kết quả đặt lại vào khung 5×5 là:

\[
NMS=\begin{bmatrix}
0&0&0&0&0\\
0&0&0&10&0\\
0&72&0&0&0\\
0&160&0&7&0\\
0&0&0&0&0
\end{bmatrix}.
\]

Ví dụ độ lớn `19`, hướng `45°`: so với `8` và `20`; vì `19<20` nên bị loại. Độ lớn `72`, hướng `45°`: so với `19` và `44`; `72` lớn nhất nên được giữ.

---

### Bài 4 — Ngưỡng kép và theo dõi cạnh theo độ trễ

Với ngưỡng thấp `50`, cao `100`, quy ước của slide:

- mạnh (`2`) nếu `I≥100`;
- yếu (`1`) nếu `50≤I<100`;
- ngoại lai (`0`) nếu `I<50`.

Sau phân loại:

\[
T=\begin{bmatrix}
0&0&0&1&0\\
1&2&1&0&0\\
0&0&2&2&0\\
0&1&1&1&0\\
0&0&0&0&0
\end{bmatrix}.
\]

Slide xử lý các pixel yếu theo một lượt trực tiếp: pixel yếu chỉ được giữ khi tại thời điểm xét đã có pixel mạnh trong 8-lân-cận. Kết quả theo slide là:

\[
H=\begin{bmatrix}
0&0&0&0&0\\
2&2&2&0&0\\
0&0&2&2&0\\
0&2&2&2&0\\
0&0&0&0&0
\end{bmatrix}.
\]

Đổi `2→255` để hiển thị ảnh nhị phân.

Lưu ý về Canny chuẩn: hysteresis thường được cài bằng DFS/BFS từ mọi pixel mạnh, nên giữ toàn bộ chuỗi pixel yếu có đường nối tới cạnh mạnh. Khi đó pixel `80` ở hàng 1 cột 4 nối qua pixel `90` tới pixel mạnh `120` và cũng được giữ; ô đầu hàng trở thành `2`. `code/chuong3.py` dùng phiên bản BFS chuẩn, không phụ thuộc thứ tự duyệt. Khi làm đúng đáp án slide dùng ma trận phía trên; khi cài thuật toán thực tế dùng kết quả BFS.

## II. Bài tập lập trình

Mã hoàn chỉnh: `code/chuong3.py`.

### Bài 1 — Đạo hàm với kernel trung bình 3 hàng

\[
K_x=\frac13\begin{bmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{bmatrix},\quad
K_y=\frac13\begin{bmatrix}1&1&1\\0&0&0\\-1&-1&-1\end{bmatrix}.
\]

Quy trình: đổi xám → tích chập → tính `magnitude=hypot(Ix,Iy)` → `angle=atan2(Iy,Ix)`. Cần giữ kiểu `float`; nếu ép sớm sang `uint8`, mọi đạo hàm âm sẽ mất.

### Bài 2 — Sobel

Giống Bài 1 nhưng trọng số hàng/cột giữa bằng 2 và toàn kernel chia 8. Mã vừa lưu `Ix/Iy` dưới dạng ảnh chuẩn hóa để nhìn, vừa lưu magnitude thực.

### Bài 3 — Canny viết từng bước

Lý do thứ tự năm bước không thể đảo:

1. **Gaussian** giảm nhiễu, vì đạo hàm khuếch đại nhiễu.
2. **Sobel** tìm độ lớn và hướng thay đổi.
3. **NMS** làm cạnh từ dải dày thành đường mảnh.
4. **Ngưỡng kép** tách chắc chắn/có thể/không phải cạnh.
5. **Hysteresis** chỉ giữ cạnh yếu có kết nối với cạnh mạnh.

Mã so sánh với `cv2.Canny`. Hai kết quả có thể không giống từng bit vì OpenCV dùng các chi tiết triển khai, chuẩn gradient và nội suy NMS khác, nhưng vị trí cạnh chính phải tương đồng.
