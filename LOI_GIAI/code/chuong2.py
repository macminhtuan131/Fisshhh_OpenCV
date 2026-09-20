"""Lời giải lập trình Chương 2: Image Basics and Filtering.
Tuấn code ngu, AI sửa vãi lìn thật :))) Dell code nữa
Chạy:
    python chuong2.py
    python chuong2.py --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DATA = WORKSPACE / "Bài tập đưa sinh viên" / "Chuong2" / "data"
OUT = HERE.parent / "output" / "chuong2"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_image(name: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Đọc ảnh và báo lỗi rõ ràng thay vì để lỗi None xuất hiện về sau."""
    path = DATA / name
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return image


def bgr_to_gray_formula(image_bgr: np.ndarray) -> np.ndarray:
    """I = 0.299R + 0.587G + 0.114B, tính bằng float rồi mới làm tròn."""
    b, g, r = cv2.split(image_bgr.astype(np.float64))
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def source_coordinate(dst_index: int, src_size: int, dst_size: int) -> float:
    """Ánh xạ căn chỉnh hai đầu biên: 0 -> 0 và dst_size-1 -> src_size-1."""
    if dst_size <= 1:
        return 0.0
    return dst_index * (src_size - 1) / (dst_size - 1)


def resize_nearest(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    """Nội suy gần nhất viết trực tiếp từ công thức ánh xạ ngược."""
    new_h, new_w = new_size
    old_h, old_w = image.shape[:2]
    result = np.empty((new_h, new_w) + image.shape[2:], dtype=image.dtype)
    for y2 in range(new_h):
        y1 = int(round(source_coordinate(y2, old_h, new_h)))
        for x2 in range(new_w):
            x1 = int(round(source_coordinate(x2, old_w, new_w)))
            result[y2, x2] = image[y1, x1]
    return result


def sample_bilinear(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """Lấy mẫu tại tọa độ thực (x, y) bằng trung bình có trọng số bốn góc."""
    h, w = image.shape[:2]
    x = float(np.clip(x, 0, w - 1))
    y = float(np.clip(y, 0, h - 1))
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    return (
        (1 - dx) * (1 - dy) * image[y0, x0]
        + dx * (1 - dy) * image[y0, x1]
        + (1 - dx) * dy * image[y1, x0]
        + dx * dy * image[y1, x1]
    )


def resize_bilinear(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    """Nội suy song tuyến viết trực tiếp, hỗ trợ ảnh xám hoặc nhiều kênh."""
    new_h, new_w = new_size
    old_h, old_w = image.shape[:2]
    source = image.astype(np.float64)
    result = np.empty((new_h, new_w) + image.shape[2:], dtype=np.float64)
    for y2 in range(new_h):
        y1 = source_coordinate(y2, old_h, new_h)
        for x2 in range(new_w):
            x1 = source_coordinate(x2, old_w, new_w)
            result[y2, x2] = sample_bilinear(source, x1, y1)
    return np.clip(np.rint(result), 0, 255).astype(image.dtype)


def rotate_bilinear(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Quay quanh tâm, giữ kích thước, ánh xạ ngược và nội suy song tuyến."""
    h, w = image.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    theta = np.deg2rad(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    source = image.astype(np.float64)
    result = np.zeros_like(source)

    # Pixel đích p_d = R p_s, do đó p_s = R^{-1} p_d.
    for yd in range(h):
        for xd in range(w):
            xr, yr = xd - cx, yd - cy
            # OpenCV dùng tọa độ ảnh có y hướng xuống. Đây là nghịch đảo
            # của ma trận getRotationMatrix2D cho góc dương.
            xs = c * xr - s * yr + cx
            ys = s * xr + c * yr + cy
            if 0 <= xs <= w - 1 and 0 <= ys <= h - 1:
                result[yd, xd] = sample_bilinear(source, xs, ys)
    return np.clip(np.rint(result), 0, 255).astype(image.dtype)


def save_figure(fig: plt.Figure, name: str, show: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def exercise_1(show: bool) -> None:
    image = read_image("C2_B1.png")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_manual = bgr_to_gray_formula(image)
    difference = cv2.absdiff(gray_cv, gray_manual)

    print(f"Bài 1: shape={image.shape}, dtype={image.dtype}")
    print(
        "  Sai số đổi xám: max =",
        int(difference.max()),
        ", mean =",
        float(difference.mean()),
    )

    b, g, r = cv2.split(image)
    zeros = np.zeros_like(b)
    panels = [
        (rgb, "Ảnh RGB", None),
        (cv2.merge([r, zeros, zeros]), "Kênh R", None),
        (cv2.merge([zeros, g, zeros]), "Kênh G", None),
        (cv2.merge([zeros, zeros, b]), "Kênh B", None),
        (gray_cv, "Gray OpenCV", "gray"),
        (gray_manual, "Gray công thức", "gray"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (data, title, cmap) in zip(axes.ravel(), panels):
        ax.imshow(data, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    save_figure(fig, "bai1_kenh_mau_va_anh_xam.png", show)


def exercise_2(show: bool) -> None:
    image = read_image("C2_B2.PNG")
    h, w = image.shape[:2]
    target = (2 * h, 2 * w)
    nearest = resize_nearest(image, target)
    bilinear = resize_bilinear(image, target)
    cv_nearest = cv2.resize(image, (2 * w, 2 * h), interpolation=cv2.INTER_NEAREST)
    cv_linear = cv2.resize(image, (2 * w, 2 * h), interpolation=cv2.INTER_LINEAR)

    # OpenCV dùng quy ước half-pixel, còn hàm tay căn chỉnh góc; vì vậy không buộc hai ảnh giống bit.
    print("Bài 2:")
    print("  Kích thước gốc/đích:", image.shape, nearest.shape)
    print("  MAE bilinear tay/OpenCV:", np.abs(bilinear.astype(float) - cv_linear).mean())

    images = [image, nearest, bilinear, cv_nearest, cv_linear]
    titles = ["Gốc", "Nearest tay", "Bilinear tay", "OpenCV nearest", "OpenCV linear"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    for ax, data, title in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    save_figure(fig, "bai2_noi_suy.png", show)


def exercise_3(show: bool) -> None:
    image = read_image("C2_B3.PNG")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = np.bincount(gray.ravel(), minlength=256)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Ảnh xám")
    axes[0].axis("off")
    axes[1].plot(np.arange(256), histogram)
    axes[1].set(xlim=(0, 255), xlabel="Mức xám", ylabel="Số pixel", title="Histogram")
    axes[1].grid(alpha=0.3)
    save_figure(fig, "bai3_histogram.png", show)


def exercise_4(show: bool) -> None:
    image = read_image("C2_B4.png")
    manual = rotate_bilinear(image, 30.0)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), 30.0, 1.0)
    opencv = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, data, title in zip(axes, [image, manual, opencv], ["Gốc", "Quay tay 30°", "OpenCV 30°"]):
        ax.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    save_figure(fig, "bai4_quay_anh.png", show)


def exercise_5(show: bool) -> None:
    image = read_image("C2_B5.PNG")
    mean = cv2.blur(image, (7, 7))
    gaussian = cv2.GaussianBlur(image, (7, 7), sigmaX=10, sigmaY=10)
    median = cv2.medianBlur(image, 7)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    titles = ["Gốc", "Trung bình 7×7", "Gaussian 7×7, σ=10", "Trung vị 7×7"]
    for ax, data, title in zip(axes, [image, mean, gaussian, median], titles):
        ax.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    save_figure(fig, "bai5_bo_loc.png", show)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Mở cửa sổ hình ngoài việc lưu file")
    args = parser.parse_args()
    exercise_1(args.show)
    exercise_2(args.show)
    exercise_3(args.show)
    exercise_4(args.show)
    exercise_5(args.show)
    print(f"Đã lưu kết quả Chương 2 vào: {OUT}")


if __name__ == "__main__":
    main()
