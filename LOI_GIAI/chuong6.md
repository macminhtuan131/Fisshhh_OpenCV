# Chương 6 — Deep Learning nhận diện chữ số MNIST

Đề có một bài lớn: tải/đọc MNIST, xây dựng ANN, huấn luyện, đánh giá, lưu mô hình và triển khai trên ảnh chữ số bên ngoài.

## 1. Nhận dạng bài toán

Mỗi ảnh MNIST là ma trận xám `28×28`, nhãn là một số nguyên từ 0 đến 9. Ta có ảnh **và** nhãn, nên đây là học có giám sát; đầu ra thuộc một trong 10 lớp, nên đây là phân loại đa lớp.

- Train: 60.000 ảnh để học tham số.
- Test: 10.000 ảnh chỉ dùng đánh giá cuối.
- Validation: tách từ train để chọn mô hình/quan sát overfitting.

Không được dùng test để điều chỉnh mô hình nhiều lần, vì khi đó test không còn là đánh giá khách quan.

## 2. Đọc định dạng IDX

Bốn file trong `Chuong6/data/input` chứa dữ liệu nhị phân big-endian:

- file ảnh: header `magic=2051, số ảnh, số hàng, số cột`, sau đó là byte pixel;
- file nhãn: header `magic=2049, số nhãn`, sau đó là byte nhãn.

Mã `read_idx_images/read_idx_labels` kiểm tra magic number và kích thước file trước khi reshape. Đây là lý do không nên chỉ gọi `reshape(-1,28,28)` mà bỏ header: 16 byte đầu không phải pixel.

## 3. Tiền xử lý

Pixel gốc nằm trong `[0,255]`. Chia cho 255:

\[
x'=x/255\in[0,1].
\]

Chuẩn hóa giúp gradient ổn định hơn và Adam/SGD chọn bước cập nhật dễ hơn. Không thay nhãn vì dùng `sparse_categorical_crossentropy`, hàm này nhận trực tiếp nhãn nguyên 0–9.

## 4. Mạng ANN hoạt động thế nào?

### Flatten

Biến ảnh `28×28` thành vector 784 phần tử. Flatten không có tham số và không “học” gì; nó chỉ đổi hình dạng.

### Dense

Một lớp đầy đủ tính:

\[
z=W x+b,\qquad a=\phi(z).
\]

Mỗi trọng số trong `W` đo mức ảnh hưởng của một đầu vào tới một neuron; `b` là độ lệch.

Số tham số Dense với `n_in` đầu vào và `n_out` neuron:

\[
n_{param}=n_{in}n_{out}+n_{out}.
\]

Vì vậy mô hình lớn trong slide `784→2048→4096→10` có:

- `784×2048+2048 = 1,607,680`;
- `2048×4096+4096 = 8,392,704`;
- `4096×10+10 = 40,970`;
- tổng `10,041,354` tham số.

Đây là lý do nó chậm và tốn RAM/VRAM.

### Hàm kích hoạt

Sigmoid:

\[
\sigma(z)=\frac1{1+e^{-z}}.
\]

Với mạng sâu/lớn, sigmoid dễ bão hòa: ở `|z|` lớn đạo hàm gần 0, gradient truyền ngược yếu. ReLU:

\[
ReLU(z)=\max(0,z)
\]

thường huấn luyện ANN ảnh nhanh hơn, nên mã mặc định dùng mô hình gọn `784→128→64→10` với ReLU.

### Softmax lớp ra

Với 10 logit `z_k`:

\[
p_k=\frac{e^{z_k}}{\sum_{j=0}^{9}e^{z_j}},\qquad \sum_kp_k=1.
\]

`p_k` được hiểu là độ tin cậy tương đối cho lớp `k`; dự đoán là `argmax(p)`.

## 5. Hàm mất mát và cập nhật trọng số

Với nhãn thật `y`, sparse categorical cross-entropy:

\[
L=-\log p_y.
\]

Nếu mô hình cho xác suất cao ở lớp đúng, `L` nhỏ; nếu tự tin vào lớp sai, `L` rất lớn. Backpropagation dùng quy tắc dây chuyền tính `∂L/∂W`; optimizer cập nhật:

\[
W\leftarrow W-\eta\frac{\partial L}{\partial W}.
\]

SGD dùng gradient hiện tại. Adam duy trì trung bình động của gradient và bình phương gradient, nên thường hội tụ nhanh hơn với ít chỉnh learning rate hơn. Đây là cơ sở để mô hình ReLU+Adam trong slide tốt/nhanh hơn sigmoid+SGD.

## 6. Epoch, batch và validation

- Batch size 64: tính một lần cập nhật từ 64 mẫu; nhanh hơn cập nhật từng mẫu và ít tốn RAM hơn toàn bộ 60.000 mẫu.
- Một epoch: mô hình đã đi qua toàn bộ tập train một lần.
- Nhiều epoch quá có thể làm train accuracy tăng nhưng validation accuracy giảm — overfitting.

Mã dùng `validation_split=0.2`, `EarlyStopping` và khôi phục trọng số validation tốt nhất.

## 7. Đánh giá đúng

`accuracy = số dự đoán đúng / tổng số mẫu`. Với từng lớp còn có:

\[
Precision=\frac{TP}{TP+FP},\qquad
Recall=\frac{TP}{TP+FN},\qquad
F1=\frac{2PR}{P+R}.
\]

Ma trận nhầm lẫn `C[i,j]` đếm số ảnh lớp thật `i` bị dự đoán thành `j`. Các ô ngoài đường chéo cho biết cặp chữ số hay nhầm, hữu ích hơn một con số accuracy duy nhất.

## 8. Lưu và triển khai

Mã lưu mô hình Keras (`.keras` hoặc `.h5`) và có thể chuyển sang TensorFlow Lite (`.tflite`). TFLite phù hợp thiết bị di động/nhúng vì runtime gọn hơn.

Ảnh ngoài MNIST phải được đưa về cùng phân phối:

1. đổi xám;
2. nếu nền sáng/chữ tối thì đảo màu;
3. tách vùng chữ số;
4. co giãn giữ tỉ lệ vào hộp khoảng 20×20;
5. đặt giữa canvas 28×28 nền đen;
6. chia 255 như khi train;
7. thêm chiều batch thành `(1,28,28)`.

Nếu chỉ `resize(28,28)` mà không đảo màu/căn giữa, mô hình có thể sai dù accuracy MNIST cao. Đây là nguyên nhân phổ biến nhất khi ảnh tự vẽ dự đoán kém.

## 9. Cách chạy

Đọc và kiểm tra dữ liệu không cần TensorFlow:

```powershell
python .\code\chuong6_mnist.py --mode inspect
```

Huấn luyện nhanh để kiểm tra pipeline:

```powershell
python .\code\chuong6_mnist.py --mode train --epochs 1 --train-limit 5000 --test-limit 1000
```

Huấn luyện đầy đủ mô hình gọn:

```powershell
python .\code\chuong6_mnist.py --mode train --architecture compact --epochs 10
```

Muốn tái hiện đúng kiến trúc slide (rất nặng):

```powershell
python .\code\chuong6_mnist.py --mode train --architecture slide --epochs 10
```

Dự đoán ảnh ngoài:

```powershell
python .\code\chuong6_mnist.py --mode predict --image "..\Bài tập đưa sinh viên\Chuong6\data\data_test\a2.png"
```

## 10. Các lỗi trong mã mẫu cần tránh

- Mã mẫu train trực tiếp pixel 0–255: chạy được nhưng tối ưu kém ổn định hơn chuẩn hóa `[0,1]`.
- `ImageProcessing(img_filename)` lại đọc biến toàn cục `img_name`; phải đọc chính tham số `img_filename`.
- `np.argmax(result)` chỉ an toàn khi batch có đúng một ảnh; tổng quát phải dùng `np.argmax(result, axis=1)`.
- Ảnh test ngoài cần cùng cách chuẩn hóa với ảnh train.
- `random.randint(1,60000)` có thể trả 60000, vượt chỉ số cuối 59999; nên dùng `randrange(60000)` hoặc generator NumPy.
- Python hiện tại của máy có thể quá mới so với TensorFlow. Dùng môi trường Python 3.10/3.11 riêng thay vì hạ/nâng gói lung tung trong môi trường chính.
