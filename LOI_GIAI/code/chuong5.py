"""Lời giải lập trình Chương 5: K-means, threshold và morphology."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DATA = WORKSPACE / "Bài tập đưa sinh viên" / "Chuong5" / "data"
OUT = HERE.parent / "output" / "chuong5"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_image(name: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    path = DATA / name
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def initialize_kmeans_pp(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++: tâm sau ưu tiên điểm xa các tâm đã chọn."""
    n = len(data)
    centers = [data[rng.integers(n)].copy()]
    for _ in range(1, k):
        squared = np.min(
            np.sum((data[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2), axis=1
        )
        total = squared.sum()
        index = rng.integers(n) if total == 0 else rng.choice(n, p=squared / total)
        centers.append(data[index].copy())
    return np.asarray(centers, dtype=np.float64)


def kmeans(
    data: np.ndarray, k: int, seed: int = 7, max_iter: int = 100, tol: float = 1e-5
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """K-means Euclid tự cài; trả labels, centers và lịch sử inertia."""
    points = np.asarray(data, dtype=np.float64)
    if points.ndim != 2 or len(points) < k:
        raise ValueError("data phải có dạng (N,D) và N >= k")
    rng = np.random.default_rng(seed)
    centers = initialize_kmeans_pp(points, k, rng)
    history: list[float] = []

    for _ in range(max_iter):
        squared = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(squared, axis=1)
        history.append(float(np.min(squared, axis=1).sum()))
        new_centers = np.empty_like(centers)
        nearest_squared = np.min(squared, axis=1)
        for cluster in range(k):
            members = points[labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
            else:
                # Cứu cụm rỗng bằng điểm hiện đang được biểu diễn tệ nhất.
                index = int(np.argmax(nearest_squared))
                new_centers[cluster] = points[index]
                nearest_squared[index] = -1
        movement = float(np.linalg.norm(new_centers - centers, axis=1).max())
        centers = new_centers
        if movement < tol:
            break

    squared = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(squared, axis=1)
    return labels, centers, history


def save_figure(fig: plt.Figure, name: str, show: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def exercise_1(show: bool) -> None:
    points = np.loadtxt(DATA / "C5_B1.txt", dtype=np.float64)
    labels, centers, history = kmeans(points, 3, seed=7)
    print("Bài 1 — tâm ba cụm:\n", centers)
    print("  Inertia theo lượt:", " -> ".join(f"{x:.3f}" for x in history))

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for cluster in range(3):
        members = points[labels == cluster]
        ax.scatter(members[:, 0], members[:, 1], s=20, color=colors[cluster], label=f"Cụm {cluster + 1}")
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=180, c="black", label="Tâm")
    ax.set(title="K-means K=3", xlabel="x", ylabel="y")
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, "bai1_kmeans_2d.png", show)


def exercise_2(show: bool) -> None:
    gray = read_image("C5_B2.PNG", cv2.IMREAD_GRAYSCALE)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    levels = np.flatnonzero(histogram > 0).astype(np.float64)
    frequency = histogram[levels.astype(int)]
    features = np.column_stack([levels, frequency])

    # Hai chiều khác đơn vị nên chuẩn hóa z-score trước K-means.
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1
    labels, _, _ = kmeans((features - mean) / std, 2, seed=11)

    weighted_gray_means = []
    for cluster in range(2):
        mask = labels == cluster
        weighted_gray_means.append(np.average(levels[mask], weights=frequency[mask]))
    weighted_gray_means = np.sort(weighted_gray_means)
    threshold = float(np.mean(weighted_gray_means))
    binary_kmeans = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    binary_fixed = np.where(gray >= 127, 255, 0).astype(np.uint8)
    print(f"Bài 2 — tâm mức xám có trọng số={weighted_gray_means}, ngưỡng={threshold:.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Ảnh xám")
    axes[0, 1].plot(histogram)
    axes[0, 1].axvline(threshold, color="red", label=f"K-means T={threshold:.1f}")
    axes[0, 1].axvline(127, color="green", linestyle="--", label="Cố định T=127")
    axes[0, 1].set(title="Histogram", xlabel="Mức xám", ylabel="Tần số")
    axes[0, 1].legend()
    axes[1, 0].imshow(binary_fixed, cmap="gray")
    axes[1, 0].set_title("Ngưỡng cố định 127")
    axes[1, 1].imshow(binary_kmeans, cmap="gray")
    axes[1, 1].set_title("Ngưỡng từ K-means")
    for ax in axes.ravel():
        if ax is not axes[0, 1]:
            ax.axis("off")
    save_figure(fig, "bai2_kmeans_histogram.png", show)


def exercise_3(show: bool) -> None:
    image = read_image("C5_B3.PNG")
    pixels = image.reshape(-1, 3).astype(np.float64)
    labels, centers, history = kmeans(pixels, 5, seed=19, max_iter=50, tol=0.1)
    segmented = np.clip(np.rint(centers[labels]), 0, 255).astype(np.uint8).reshape(image.shape)
    print("Bài 3 — tâm màu BGR:\n", np.rint(centers).astype(int))
    print("  Số lượt:", len(history))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, data, title in zip(axes, [image, segmented], ["Ảnh gốc", "K-means màu K=5"]):
        ax.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    save_figure(fig, "bai3_kmeans_mau.png", show)


def morphology_manual(binary: np.ndarray, kernel: np.ndarray, operation: str) -> np.ndarray:
    """Dilation/erosion nhị phân tự cài với biên ngoài bằng 0."""
    source = binary > 0
    footprint = np.asarray(kernel, dtype=bool)
    kh, kw = footprint.shape
    py, px = kh // 2, kw // 2
    padded = np.pad(source, ((py, py), (px, px)), mode="constant", constant_values=False)
    result = np.zeros_like(source)
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            selected = padded[y : y + kh, x : x + kw][footprint]
            result[y, x] = np.any(selected) if operation == "dilate" else np.all(selected)
    return (result.astype(np.uint8) * 255)


def exercise_4(show: bool) -> None:
    gray = read_image("C5_B4.jpg", cv2.IMREAD_GRAYSCALE)
    binary = np.where(gray >= 150, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilation = morphology_manual(binary, kernel, "dilate")
    erosion = morphology_manual(binary, kernel, "erode")
    opening = morphology_manual(erosion, kernel, "dilate")
    closing = morphology_manual(morphology_manual(binary, kernel, "dilate"), kernel, "erode")

    # Đối chiếu nội bộ với OpenCV cùng quy ước biên 0.
    cv_dilation = cv2.dilate(binary, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    cv_erosion = cv2.erode(binary, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    print("Bài 4 — dilation khớp OpenCV:", np.array_equal(dilation, cv_dilation))
    print("         erosion khớp OpenCV:", np.array_equal(erosion, cv_erosion))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        (gray, "Xám"),
        (binary, "Threshold T=150"),
        (dilation, "Dilation"),
        (erosion, "Erosion"),
        (opening, "Opening"),
        (closing, "Closing"),
    ]
    for ax, (data, title) in zip(axes.ravel(), panels):
        ax.imshow(data, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    save_figure(fig, "bai4_morphology.png", show)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    exercise_1(args.show)
    exercise_2(args.show)
    exercise_3(args.show)
    exercise_4(args.show)
    print(f"Đã lưu kết quả Chương 5 vào: {OUT}")


if __name__ == "__main__":
    main()
