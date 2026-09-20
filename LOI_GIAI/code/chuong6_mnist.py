"""Lời giải Chương 6: đọc IDX, huấn luyện, đánh giá và triển khai MNIST.

Phần ``--mode inspect`` chỉ cần NumPy/Matplotlib.
Phần train/predict cần TensorFlow trong môi trường Python được hỗ trợ.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
CHAPTER_DATA = WORKSPACE / "Bài tập đưa sinh viên" / "Chuong6" / "data"
INPUT = CHAPTER_DATA / "input"
OUT = HERE.parent / "output" / "chuong6"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN_IMAGES = INPUT / "train-images-idx3-ubyte" / "train-images.idx3-ubyte"
TRAIN_LABELS = INPUT / "train-labels-idx1-ubyte" / "train-labels.idx1-ubyte"
TEST_IMAGES = INPUT / "t10k-images-idx3-ubyte" / "t10k-images.idx3-ubyte"
TEST_LABELS = INPUT / "t10k-labels-idx1-ubyte" / "t10k-labels.idx1-ubyte"


def read_idx_images(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 16:
        raise ValueError(f"File ảnh IDX quá ngắn: {path}")
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"Magic ảnh phải là 2051, nhận {magic}: {path}")
    expected = 16 + count * rows * cols
    if len(raw) != expected:
        raise ValueError(f"Sai kích thước {path}: cần {expected} byte, có {len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, rows, cols).copy()


def read_idx_labels(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"File nhãn IDX quá ngắn: {path}")
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"Magic nhãn phải là 2049, nhận {magic}: {path}")
    if len(raw) != 8 + count:
        raise ValueError(f"Sai kích thước {path}: cần {8 + count} byte, có {len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8, offset=8).copy()


def load_local_mnist() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    x_train, y_train = read_idx_images(TRAIN_IMAGES), read_idx_labels(TRAIN_LABELS)
    x_test, y_test = read_idx_images(TEST_IMAGES), read_idx_labels(TEST_LABELS)
    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("Số ảnh và số nhãn không khớp")
    return (x_train, y_train), (x_test, y_test)


def save_samples(images: np.ndarray, labels: np.ndarray, show: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(images), size=15, replace=False)
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for ax, index in zip(axes.ravel(), indices):
        ax.imshow(images[index], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Nhãn {labels[index]}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "mnist_samples.png", dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def inspect_data(show: bool) -> None:
    (x_train, y_train), (x_test, y_test) = load_local_mnist()
    print("x_train:", x_train.shape, x_train.dtype, "range", (x_train.min(), x_train.max()))
    print("y_train:", y_train.shape, "phân bố", np.bincount(y_train, minlength=10))
    print("x_test :", x_test.shape, x_test.dtype, "range", (x_test.min(), x_test.max()))
    print("y_test :", y_test.shape, "phân bố", np.bincount(y_test, minlength=10))
    save_samples(x_train, y_train, show)


def require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "Chưa có TensorFlow. Hãy tạo môi trường Python 3.10/3.11 và chạy "
            "`python -m pip install tensorflow` trước."
        ) from exc
    return tf


def build_model(tf, architecture: str):
    if architecture == "slide":
        # Đúng mô hình 10,041,354 tham số trong slide.
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(28, 28)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2048, activation="sigmoid"),
                tf.keras.layers.Dense(4096, activation="sigmoid"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ],
            name="mnist_slide",
        )
        optimizer = "sgd"
    else:
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(28, 28)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ],
            name="mnist_compact",
        )
        optimizer = "adam"
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_history(history, show: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history.history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history.history["accuracy"], "o-", label="train")
    axes[0].plot(epochs, history.history["val_accuracy"], "x--", label="validation")
    axes[0].set(title="Accuracy", xlabel="Epoch")
    axes[1].plot(epochs, history.history["loss"], "o-", label="train")
    axes[1].plot(epochs, history.history["val_loss"], "x--", label="validation")
    axes[1].set(title="Loss", xlabel="Epoch")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "training_history.png", dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def confusion_matrix(true: np.ndarray, predicted: np.ndarray, classes: int = 10) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (true, predicted), 1)
    return matrix


def save_confusion(matrix: np.ndarray, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xlabel="Dự đoán", ylabel="Nhãn thật", title="Confusion matrix")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for row in range(10):
        for col in range(10):
            if matrix[row, col]:
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def train(args: argparse.Namespace) -> None:
    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)
    (x_train, y_train), (x_test, y_test) = load_local_mnist()
    if args.train_limit:
        x_train, y_train = x_train[: args.train_limit], y_train[: args.train_limit]
    if args.test_limit:
        x_test, y_test = x_test[: args.test_limit], y_test[: args.test_limit]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    model = build_model(tf, args.architecture)
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
        )
    ]
    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.perf_counter() - start
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(y_test, predicted)

    OUT.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model) if args.model else OUT / f"mnist_{args.architecture}.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Thời gian train: {elapsed:.2f} s")
    print(f"Test loss={loss:.4f}, accuracy={accuracy:.4f}")
    print("Đã lưu model:", model_path)
    plot_history(history, args.show)
    save_confusion(matrix, args.show)

    if args.export_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_path = model_path.with_suffix(".tflite")
        tflite_path.write_bytes(converter.convert())
        print("Đã xuất TensorFlow Lite:", tflite_path)


def preprocess_external_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Đưa chữ số ngoài về nền đen, chữ trắng, căn giữa trên canvas 28×28."""
    gray = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    normalized_polarity = 255 - gray if gray.mean() > 127 else gray.copy()
    # Ảnh mẫu có thể dùng nét màu (ví dụ xanh lá chuyển xám chỉ khoảng 150).
    # Kéo tương phản để nét sáng đạt 255 giống MNIST thay vì giữ nét quá tối.
    normalized_polarity = cv2.normalize(normalized_polarity, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(normalized_polarity, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coordinates = cv2.findNonZero(mask)
    if coordinates is None:
        raise ValueError(f"Không tìm thấy nét chữ số trong {path}")
    x, y, w, h = cv2.boundingRect(coordinates)
    crop = normalized_polarity[y : y + h, x : x + w]
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x0, y0 = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    # Dịch trọng tâm nét về giữa ảnh giống cách căn giữa phổ biến cho MNIST.
    moments = cv2.moments(canvas)
    if moments["m00"]:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        transform = np.float32([[1, 0, 13.5 - center_x], [0, 1, 13.5 - center_y]])
        canvas = cv2.warpAffine(canvas, transform, (28, 28), borderValue=0)
    if canvas.max() > 0:
        canvas = np.clip(np.rint(canvas.astype(np.float32) * 255.0 / canvas.max()), 0, 255).astype(np.uint8)
    tensor = canvas.astype(np.float32) / 255.0
    return gray, tensor


def predict(args: argparse.Namespace) -> None:
    tf = require_tensorflow()
    model_path = Path(args.model) if args.model else OUT / "mnist_compact.keras"
    if not args.image:
        raise SystemExit("Mode predict cần --image đường_dẫn_ảnh")
    original, tensor = preprocess_external_image(Path(args.image))
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(tensor[None, ...], verbose=0)[0]
    predicted = int(np.argmax(probabilities))
    print(f"Dự đoán: {predicted}, độ tin cậy softmax: {probabilities[predicted]:.4f}")
    print("Xác suất 0..9:", np.array2string(probabilities, precision=4, suppress_small=True))

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Ảnh gốc")
    axes[1].imshow(tensor, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Đầu vào 28×28")
    axes[2].bar(np.arange(10), probabilities)
    axes[2].set(xticks=np.arange(10), ylim=(0, 1), title=f"Dự đoán = {predicted}")
    for ax in axes[:2]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "external_prediction.png", dpi=160, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inspect", "train", "predict"], default="inspect")
    parser.add_argument("--architecture", choices=["compact", "slide"], default="compact")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-limit", type=int, default=0, help="0 = dùng toàn bộ")
    parser.add_argument("--test-limit", type=int, default=0, help="0 = dùng toàn bộ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", help="Đường dẫn model lưu hoặc đọc")
    parser.add_argument("--image", help="Ảnh chữ số ngoài cho mode predict")
    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "inspect":
        inspect_data(args.show)
    elif args.mode == "train":
        train(args)
    else:
        predict(args)
    print(f"Kết quả Chương 6 nằm tại: {OUT}")


if __name__ == "__main__":
    main()
