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

```text
x_normalized = x / 255, nên x_normalized nằm trong khoảng [0, 1].
```

Chuẩn hóa giúp gradient ổn định hơn và Adam/SGD chọn bước cập nhật dễ hơn. Không thay nhãn vì dùng `sparse_categorical_crossentropy`, hàm này nhận trực tiếp nhãn nguyên 0–9.

## 4. Mạng ANN hoạt động thế nào?

### Flatten

Biến ảnh `28×28` thành vector 784 phần tử. Flatten không có tham số và không “học” gì; nó chỉ đổi hình dạng.

### Dense

Một lớp đầy đủ tính:

```text
z = W*x + b
a = activation(z)
```

Mỗi trọng số trong `W` đo mức ảnh hưởng của một đầu vào tới một neuron; `b` là độ lệch.

Số tham số Dense với `n_in` đầu vào và `n_out` neuron:

```text
số tham số = n_in * n_out + n_out
```

Vì vậy mô hình lớn trong slide `784→2048→4096→10` có:

- `784×2048+2048 = 1,607,680`;
- `2048×4096+4096 = 8,392,704`;
- `4096×10+10 = 40,970`;
- tổng `10,041,354` tham số.

Đây là lý do nó chậm và tốn RAM/VRAM.

### Hàm kích hoạt

Sigmoid:

```text
sigmoid(z) = 1 / (1 + e^(-z))
```

Với mạng sâu/lớn, sigmoid dễ bão hòa: ở `|z|` lớn đạo hàm gần 0, gradient truyền ngược yếu. ReLU:

```text
ReLU(z) = max(0, z)
```

thường huấn luyện ANN ảnh nhanh hơn, nên mã mặc định dùng mô hình gọn `784→128→64→10` với ReLU.

### Softmax lớp ra

Với 10 logit `z_k`:

```text
p[k] = e^(z[k]) / tổng(e^(z[j])) với j chạy từ 0 đến 9
tổng tất cả p[k] = 1
```

`p_k` được hiểu là độ tin cậy tương đối cho lớp `k`; dự đoán là `argmax(p)`.

## 5. Hàm mất mát và cập nhật trọng số

Với nhãn thật `y`, sparse categorical cross-entropy:

```text
loss = -log(xác suất mô hình gán cho nhãn đúng y)
```

Nếu mô hình cho xác suất cao ở lớp đúng, `L` nhỏ; nếu tự tin vào lớp sai, `L` rất lớn. Backpropagation dùng quy tắc dây chuyền tính `∂L/∂W`; optimizer cập nhật:

```text
W mới = W cũ - learning_rate * gradient_của_loss_theo_W
```

SGD dùng gradient hiện tại. Adam duy trì trung bình động của gradient và bình phương gradient, nên thường hội tụ nhanh hơn với ít chỉnh learning rate hơn. Đây là cơ sở để mô hình ReLU+Adam trong slide tốt/nhanh hơn sigmoid+SGD.

## 6. Epoch, batch và validation

- Batch size 64: tính một lần cập nhật từ 64 mẫu; nhanh hơn cập nhật từng mẫu và ít tốn RAM hơn toàn bộ 60.000 mẫu.
- Một epoch: mô hình đã đi qua toàn bộ tập train một lần.
- Nhiều epoch quá có thể làm train accuracy tăng nhưng validation accuracy giảm — overfitting.

Mã dùng `validation_split=0.2`, `EarlyStopping` và khôi phục trọng số validation tốt nhất.

## 7. Đánh giá đúng

`accuracy = số dự đoán đúng / tổng số mẫu`. Với từng lớp còn có:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

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

## 11. Code và lệnh quan trọng của chương 6

### 1. Kiểm tra shape trước khi xây dựng mạng

```python
print(x_train.shape)     # (60000, 28, 28)
print(y_train.shape)     # (60000,)
print(x_test.shape)      # (10000, 28, 28)
print(x_train.dtype)     # uint8 trước chuẩn hóa
```

Luôn kiểm tra `shape`, `dtype`, `min()` và `max()` trước khi train để phát hiện dữ liệu sai sớm.

### 2. Chuẩn hóa dữ liệu

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(x_train.min(), x_train.max())       # phải gần 0.0 và 1.0
```

Phải áp dụng cùng một cách tiền xử lý cho train, test và ảnh triển khai.

### 3. Xây dựng ANN bằng Sequential

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.summary()
```

Danh sách trong `Sequential([...])` biểu diễn thứ tự dữ liệu đi qua các lớp. Lớp cuối có 10 neuron vì có 10 chữ số.

### 4. Compile mô hình

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

Dùng `sparse_categorical_crossentropy` khi nhãn là số nguyên 0–9. Nếu nhãn đã one-hot mới dùng `categorical_crossentropy`.

### 5. Huấn luyện

```python
history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
)
```

- `epochs=10`: đi qua tập train tối đa 10 lần.
- `batch_size=64`: mỗi lần cập nhật dùng 64 ảnh.
- `validation_split=0.2`: dành 20% train để kiểm tra trong lúc học.
- Kết quả `history.history` chứa loss/accuracy theo từng epoch.

### 6. Early stopping

```python
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    callbacks=[callback],
)
```

`patience=2` nghĩa là chờ thêm 2 epoch không cải thiện rồi dừng.

### 7. Đánh giá và dự đoán

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)

probabilities = model.predict(x_test[:10])
predicted_labels = np.argmax(probabilities, axis=1)

print("Nhãn thật:", y_test[:10])
print("Dự đoán:", predicted_labels)
```

`axis=1` tìm lớp có xác suất lớn nhất cho **từng ảnh**. Thiếu `axis=1` sẽ tìm một cực đại cho toàn bộ batch.

### 8. Lưu và tải mô hình

```python
model.save("mnist_compact.keras")
loaded_model = tf.keras.models.load_model("mnist_compact.keras")
```

### 9. Chuẩn bị một ảnh ngoài cho dự đoán

```python
gray = cv2.imread("a2.png", cv2.IMREAD_GRAYSCALE)
gray = 255 - gray                         # nếu ảnh có nền trắng, nét tối
gray = cv2.resize(gray, (28, 28))
tensor = gray.astype("float32") / 255.0
batch = np.expand_dims(tensor, axis=0)    # (28,28) -> (1,28,28)

probability = loaded_model.predict(batch)
digit = int(np.argmax(probability, axis=1)[0])
print("Kết quả:", digit)
```

Mã hoàn chỉnh còn cắt vùng chữ số, giữ tỉ lệ và căn trọng tâm; đoạn trên chỉ minh họa các lệnh thiết yếu.

### 10. TensorFlow Lite

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("mnist_compact.tflite", "wb") as file:
    file.write(tflite_model)
```

`"wb"` nghĩa là mở file để ghi dữ liệu nhị phân.

### 11. Các lệnh môi trường và chạy

```powershell
conda create -n mnist python=3.10 -y
conda activate mnist
python -m pip install tensorflow numpy matplotlib opencv-python

cd LOI_GIAI
python .\code\chuong6_mnist.py --mode inspect
python .\code\chuong6_mnist.py --mode train --architecture compact --epochs 10
python .\code\chuong6_mnist.py --mode predict --model .\output\chuong6\mnist_compact.keras --image "đường_dẫn_ảnh.png"
```
