"""Lời giải lập trình Chương 3: đạo hàm, Sobel và Canny từng bước."""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DATA = WORKSPACE / "Bài tập đưa sinh viên" / "Chuong3" / "data"
OUT = HERE.parent / "output" / "chuong3"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_gray(name: str) -> np.ndarray:
    path = DATA / name
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Tích chập: lật kernel 180°, rồi dùng filter2D để tính tương quan."""
    flipped = np.flip(np.asarray(kernel, dtype=np.float64), axis=(0, 1))
    return cv2.filter2D(image.astype(np.float64), cv2.CV_64F, flipped, borderType=cv2.BORDER_CONSTANT)


def gradient(image: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> tuple[np.ndarray, ...]:
    ix = convolve(image, kx)
    iy = convolve(image, ky)
    magnitude = np.hypot(ix, iy)
    angle = np.rad2deg(np.arctan2(iy, ix))
    return ix, iy, magnitude, angle


def display_signed(values: np.ndarray) -> np.ndarray:
    """Đưa zero về xám giữa để nhìn được cả đạo hàm âm và dương."""
    maximum = float(np.max(np.abs(values)))
    if maximum == 0:
        return np.full(values.shape, 127, dtype=np.uint8)
    return np.clip(np.rint(127.5 + 127.5 * values / maximum), 0, 255).astype(np.uint8)


def display_positive(values: np.ndarray) -> np.ndarray:
    maximum = float(values.max())
    if maximum == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip(np.rint(255 * values / maximum), 0, 255).astype(np.uint8)


def non_maximum_suppression(magnitude: np.ndarray, angle_degrees: np.ndarray) -> np.ndarray:
    """Giữ cực đại địa phương theo một trong bốn hướng gradient."""
    h, w = magnitude.shape
    result = np.zeros_like(magnitude)
    angle = angle_degrees % 180.0  # cạnh không phân biệt hướng ngược 180°

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            a = angle[y, x]
            if a < 22.5 or a >= 157.5:       # 0°: trái/phải
                q, r = magnitude[y, x + 1], magnitude[y, x - 1]
            elif a < 67.5:                   # 45°
                q, r = magnitude[y - 1, x + 1], magnitude[y + 1, x - 1]
            elif a < 112.5:                  # 90°: trên/dưới
                q, r = magnitude[y - 1, x], magnitude[y + 1, x]
            else:                            # 135°
                q, r = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]
            if magnitude[y, x] >= q and magnitude[y, x] >= r:
                result[y, x] = magnitude[y, x]
    return result


def double_threshold(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """0=ngoại lai, 1=yếu, 2=mạnh."""
    if not 0 <= low <= high:
        raise ValueError("Cần 0 <= low <= high")
    labels = np.zeros(values.shape, dtype=np.uint8)
    labels[(values >= low) & (values < high)] = 1
    labels[values >= high] = 2
    return labels


def hysteresis(labels: np.ndarray) -> np.ndarray:
    """Lan từ mọi pixel mạnh qua chuỗi pixel yếu theo 8-lân-cận."""
    h, w = labels.shape
    result = np.zeros_like(labels, dtype=np.uint8)
    queue: deque[tuple[int, int]] = deque(map(tuple, np.argwhere(labels == 2)))
    for y, x in queue:
        result[y, x] = 255

    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if labels[ny, nx] == 1 and result[ny, nx] == 0:
                    result[ny, nx] = 255
                    queue.append((ny, nx))
    return result


def manual_canny(image: np.ndarray, low: float = 50, high: float = 100) -> dict[str, np.ndarray]:
    smooth = cv2.GaussianBlur(image, (5, 5), sigmaX=1.4).astype(np.float64)
    # Dấu kernel không ảnh hưởng magnitude và NMS vì góc được xét modulo 180°.
    sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    ix, iy, mag, angle = gradient(smooth, sx, sy)
    nms = non_maximum_suppression(mag, angle)
    labels = double_threshold(nms, low, high)
    edges = hysteresis(labels)
    return {
        "smooth": smooth,
        "ix": ix,
        "iy": iy,
        "magnitude": mag,
        "angle": angle,
        "nms": nms,
        "labels": labels,
        "edges": edges,
    }


def save_gradient_exercise(name: str, kx: np.ndarray, ky: np.ndarray, output_name: str, show: bool) -> None:
    image = read_gray(name)
    ix, iy, mag, angle = gradient(image, kx, ky)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    panels = [
        (image, "Ảnh xám", "gray"),
        (display_signed(ix), "Ix (âm/xám/dương)", "gray"),
        (display_signed(iy), "Iy (âm/xám/dương)", "gray"),
        (display_positive(mag), "Độ lớn", "gray"),
        (angle, "Hướng (độ)", "hsv"),
    ]
    for ax, (data, title, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / output_name, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"{output_name}: |gradient|max={mag.max():.2f}")


def exercise_1_and_2(show: bool) -> None:
    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float) / 3
    ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=float) / 3
    save_gradient_exercise("C3_B1.png", kx, ky, "bai1_gradient.png", show)

    sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float) / 8
    sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float) / 8
    save_gradient_exercise("C3_B2.png", sx, sy, "bai2_sobel.png", show)


def exercise_3(show: bool) -> None:
    image = read_gray("C3_B3.png")
    stages = manual_canny(image, low=50, high=100)
    opencv = cv2.Canny(cv2.GaussianBlur(image, (5, 5), 1.4), 50, 100)

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    panels = [
        (image, "Gốc", "gray"),
        (stages["smooth"], "1. Gaussian", "gray"),
        (display_signed(stages["ix"]), "2a. Ix", "gray"),
        (display_signed(stages["iy"]), "2b. Iy", "gray"),
        (display_positive(stages["magnitude"]), "2c. Magnitude", "gray"),
        (display_positive(stages["nms"]), "3. NMS", "gray"),
        (stages["edges"], "4–5. Ngưỡng + hysteresis", "gray"),
        (opencv, "cv2.Canny", "gray"),
    ]
    for ax, (data, title, cmap) in zip(axes.ravel(), panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / "bai3_canny_tung_buoc.png", dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print("Bài 3: số pixel cạnh tay/OpenCV =", np.count_nonzero(stages["edges"]), np.count_nonzero(opencv))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    exercise_1_and_2(args.show)
    exercise_3(args.show)
    print(f"Đã lưu kết quả Chương 3 vào: {OUT}")


if __name__ == "__main__":
    main()
