# Chương 5 — Segmentation

## I. Bài tập tính toán

### Bài 1 — K-means hai cụm

Các điểm:

`p1=(1,1), p2=(1,2), p3=(2,1), p4=(2,2), p5=(5,5), p6=(5,6), p7=(6,5), p8=(6,6), p9=(4.5,4.5), p10=(0.5,0.5)`.

Tâm đầu: `C1=(1,6)`, `C2=(6,1)`.

#### Mấu chốt của K-means

K-means tối thiểu hóa tổng bình phương khoảng cách từ điểm tới tâm cụm:

\[
J=\sum_i\|p_i-c_{label(i)}\|_2^2.
\]

Vì vậy hai bước phải lặp:

1. Gán mỗi điểm vào tâm gần nhất theo
   \[
   d(p,c)=\sqrt{(x_p-x_c)^2+(y_p-y_c)^2}.
   \]
2. Cập nhật mỗi tâm bằng trung bình các điểm trong cụm:
   \[
   c_j=\frac1{|S_j|}\sum_{p_i\in S_j}p_i.
   \]

Nếu khoảng cách hòa, kết quả phụ thuộc quy tắc phá hòa. Slide gán hòa vào `C2` ở lượt đầu.

**Lượt 1 theo slide:**

- `C1`: `p2,p6` → tâm mới `(3,4)`.
- `C2`: các điểm còn lại → tâm mới `(3.375,3.125)`.

**Lượt 2:** sau khi tính lại khoảng cách:

- cụm thấp: `p1,p2,p3,p4,p10` → tâm `(1.3,1.3)`;
- cụm cao: `p5,p6,p7,p8,p9` → tâm `(5.3,5.3)`.

**Lượt 3:** phép gán không đổi, tâm vẫn `(1.3,1.3)` và `(5.3,5.3)` nên dừng.

Kết quả cuối:

\[
S_1=\{p_1,p_2,p_3,p_4,p_{10}\},\quad c_1=(1.3,1.3),
\]

\[
S_2=\{p_5,p_6,p_7,p_8,p_9\},\quad c_2=(5.3,5.3).
\]

Tên nhãn cụm có thể bị hoán đổi mà nghiệm vẫn tương đương; ý nghĩa nằm ở các nhóm điểm, không nằm ở số 1/2.

---

### Bài 2 — Dilation với hai phần tử cấu trúc

Quy ước pixel đen là 1, trắng là 0:

\[
I=\begin{bmatrix}
0&0&0&0&0\\
0&1&0&0&0\\
0&1&1&1&0\\
0&0&0&1&0\\
0&0&0&0&0
\end{bmatrix}.
\]

Hai phần tử cấu trúc:

\[
F_1=\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix},\qquad
F_2=\begin{bmatrix}0&1&0\\1&1&1\\0&1&0\end{bmatrix}.
\]

#### Vì sao dilation “nở” vật thể?

Tại mỗi tâm, đầu ra bằng 1 nếu có **ít nhất một** vị trí mà ảnh và phần tử cấu trúc cùng bằng 1. Tương đương, mỗi pixel 1 của ảnh đóng dấu hình `F` quanh nó. Đệm ngoài ảnh bằng 0.

Kết quả với khối vuông:

\[
I\oplus F_1=\begin{bmatrix}
1&1&1&0&0\\
1&1&1&1&1\\
1&1&1&1&1\\
1&1&1&1&1\\
0&0&1&1&1
\end{bmatrix}.
\]

Kết quả với hình chữ thập:

\[
I\oplus F_2=\begin{bmatrix}
0&1&0&0&0\\
1&1&1&1&0\\
1&1&1&1&1\\
0&1&1&1&1\\
0&0&0&1&0
\end{bmatrix}.
\]

`F1` nở cả chéo; `F2` chỉ nở bốn hướng chính, cho thấy hình phần tử cấu trúc quyết định hình học kết quả.

## II. Bài tập lập trình

Mã hoàn chỉnh: `code/chuong5.py`.

### Bài 1 — K-means dữ liệu 2D thành ba cụm

Đọc `C5_B1.txt` bằng `np.loadtxt`. Mã cài trực tiếp vòng lặp gán/cập nhật và in inertia mỗi lượt. Điều kiện dừng là dịch chuyển tâm nhỏ hơn `tol`, không nên so sánh số thực bằng đúng tuyệt đối.

Một cụm rỗng có thể xảy ra nếu khởi tạo xấu; mã xử lý bằng cách lấy điểm xa tâm gần nhất làm tâm mới. `seed` cố định giúp tái lập kết quả.

### Bài 2 — Phân ngưỡng từ K-means trên histogram

Mỗi mức xám `g` tạo điểm hai chiều `(g, h[g])`. Hai chiều có đơn vị/range rất khác nên cần chuẩn hóa trước khi tính khoảng cách; nếu không, tần số lớn có thể lấn át độ xám hoặc ngược lại.

Sau khi phân hai cụm, sắp các tâm theo thành phần mức xám. Ngưỡng được chọn ở trung điểm giữa mức xám trung bình của hai cụm:

\[
T=\frac{\mu_{dark}+\mu_{bright}}2.
\]

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
