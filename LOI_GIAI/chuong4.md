# Chương 4 — Feature Detection

## I. Bài tập tính toán

### Bài 1 — Phân biệt góc và cạnh bằng ma trận cấu trúc

#### Vì sao trị riêng nhận ra góc?

Trong cửa sổ `W`, lập

\[
H=\begin{bmatrix}
\sum I_x^2&\sum I_xI_y\\
\sum I_xI_y&\sum I_y^2
\end{bmatrix}.
\]

`H` mô tả mức thay đổi cường độ khi dịch cửa sổ theo mọi hướng. Hai trị riêng là mức thay đổi theo hai phương chính:

- cả hai nhỏ: vùng phẳng;
- một lớn, một nhỏ: cạnh — đổi mạnh chỉ theo phương vuông góc cạnh;
- cả hai lớn: góc — dịch theo hướng nào cũng thay đổi mạnh.

Đạo hàm dùng

\[
K_x=\frac13\begin{bmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{bmatrix},\quad
K_y=\frac13\begin{bmatrix}1&1&1\\0&0&0\\-1&-1&-1\end{bmatrix}.
\]

**Điểm 1:** trong cửa sổ 3×3, slide tính được

\[
H_1=\begin{bmatrix}158950&57800\\57800&158950\end{bmatrix}.
\]

Với ma trận đối xứng dạng `[[a,b],[b,a]]`, trị riêng là `a-b` và `a+b`:

\[
\lambda_1=101150,\qquad\lambda_2=216750.
\]

Cả hai lớn ⇒ điểm 1 là **góc**.

**Điểm 2:**

\[
H_2=\begin{bmatrix}0&0\\0&390150\end{bmatrix},\qquad
(\lambda_1,\lambda_2)=(0,390150).
\]

Một nhỏ, một lớn ⇒ điểm 2 nằm trên **cạnh**.

---

### Bài 2 — Histogram of Oriented Gradients (HOG)

Độ lớn:

\[
M=\begin{bmatrix}20&30&60&90\\45&60&75&50\end{bmatrix},
\]

hướng:

\[
\Theta=\begin{bmatrix}135&67.5&150&330\\210&240&300&22.5\end{bmatrix}^{\circ}.
\]

Các bin: `0,45,90,135,180,225,270,315°`.

#### Mấu chốt nội suy phiếu bầu

Nếu `θi < θ < θi+1`, một pixel độ lớn `a` không bỏ toàn bộ vào một bin mà chia tuyến tính:

\[
\Delta h_i=\frac{\theta_{i+1}-\theta}{45}a,\qquad
\Delta h_{i+1}=\frac{\theta-\theta_i}{45}a.
\]

Ví dụ `(a,θ)=(30,67.5°)` nằm đúng giữa `45°` và `90°`, nên mỗi bin nhận 15. Góc là đại lượng vòng tròn: `330°` nằm giữa `315°` và `360°≡0°`, nên chia 60 cho bin 315° và 30 cho bin 0°.

Cộng tám pixel:

\[
h=[55,40,15,60,35,70,45,110].
\]

Chuẩn hóa L2 để giảm ảnh hưởng của độ sáng/độ tương phản chung:

\[
\tilde h=\frac{h}{\sqrt{\sum h_i^2}}
\approx[0.32,0.24,0.09,0.35,0.21,0.41,0.27,0.65].
\]

---

### Bài 3 — Chuẩn hóa vector và SSD

\[
A=[1,2,3,4,5],\quad B=[2,3,6,9,10],\quad C=[2.5,6,7,9,12].
\]

Nếu so trực tiếp, vector có độ lớn lớn hơn dễ tạo khoảng cách lớn dù “hình dạng” tương tự. Vì vậy chuẩn hóa L2:

\[
\tilde A=A/\|A\|_2,\qquad \|A\|_2=\sqrt{\sum_i A_i^2}.
\]

Kết quả:

- `Ã ≈ [0.13,0.27,0.40,0.54,0.67]`, `||A||≈7.42`;
- `B̃ ≈ [0.13,0.20,0.40,0.59,0.66]`, `||B||≈15.17`;
- `C̃ ≈ [0.14,0.34,0.39,0.51,0.67]`, `||C||≈17.78`.

Khoảng cách tổng bình phương:

\[
SSD(P,Q)=\sum_i(P_i-Q_i)^2.
\]

Theo số đã làm tròn trong slide:

\[
SSD(A,B)=0.0084,\quad SSD(A,C)=0.0058,\quad SSD(B,C)=0.027.
\]

Nhỏ nhất là `SSD(A,C)`, nên **A và C giống nhau nhất**. Nếu dùng toàn bộ chữ số thực, số lẻ hơi khác nhưng thứ tự không đổi.

---

### Bài 4 — Difference of Gaussians và keypoint SIFT

Với các ảnh làm mờ `L_s=G(σ_s)*I`, lập các ảnh sai khác kề nhau:

\[
D_s=L_{s+1}-L_s.
\]

Ví dụ ô trên-trái giữa `s=-1` và `s=0`:

\[
D_{-1}(0,0)=24.15-19.15=5.00.
\]

Lặp lại cho mọi ô tạo 5 ma trận DoG. Một ứng viên là keypoint nếu lớn hơn **hoặc nhỏ hơn** toàn bộ 26 hàng xóm: 8 cùng tầng, 9 tầng trên và 9 tầng dưới. Phải tìm cả cực đại và cực tiểu vì blob sáng và blob tối đều quan trọng.

Theo đáp án minh họa của slide, giá trị `-12.44` ở hàng 2, cột 2 của ma trận `D_{-1}=L_0-L_{-1}` được đánh dấu là keypoint vì là cực tiểu cục bộ nổi bật.

Lưu ý học thuật quan trọng: SIFT nghiêm ngặt chỉ kiểm tra tầng DoG có đủ một tầng trên và một tầng dưới. `D_{-1}` đang ở biên của bộ dữ liệu đã cho nên chưa đủ 26 hàng xóm theo chiều scale; muốn xác nhận nghiêm ngặt cần thêm một ảnh DoG trước nó. Với các tầng nội bộ hiện có và điều kiện so sánh nghiêm ngặt, không có cực trị 26-lân-cận. Đây là điểm chưa nhất quán trong slide; khi làm theo đáp án môn học dùng vị trí `-12.44`, khi cài SIFT chuẩn phải áp dụng điều kiện ba tầng.

## II. Bài tập lập trình

Mã hoàn chỉnh: `code/chuong4.py`.

### Bài 1 — Harris corner detector

Harris tránh phải tính trị riêng tại mọi pixel bằng đáp ứng:

\[
R=\det(H)-k\operatorname{trace}(H)^2
=(S_{xx}S_{yy}-S_{xy}^2)-k(S_{xx}+S_{yy})^2.
\]

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
