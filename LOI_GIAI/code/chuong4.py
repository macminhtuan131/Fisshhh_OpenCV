"""Lời giải lập trình Chương 4: Harris và HOG viết từ công thức."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DATA = WORKSPACE / "Bài tập đưa sinh viên" / "Chuong4" / "data"
OUT = HERE.parent / "output" / "chuong4"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_color(name: str) -> np.ndarray:
    path = DATA / name
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def harris_response(gray: np.ndarray, window: int = 5, k: float = 0.04) -> np.ndarray:
    """R = det(H) - k trace(H)^2 cho toàn ảnh bằng phép cộng cửa sổ."""
    source = gray.astype(np.float64) / 255.0
    ix = cv2.Sobel(source, cv2.CV_64F, 1, 0, ksize=3)
    iy = cv2.Sobel(source, cv2.CV_64F, 0, 1, ksize=3)
    sxx = cv2.boxFilter(ix * ix, -1, (window, window), normalize=False)
    syy = cv2.boxFilter(iy * iy, -1, (window, window), normalize=False)
    sxy = cv2.boxFilter(ix * iy, -1, (window, window), normalize=False)
    determinant = sxx * syy - sxy * sxy
    trace = sxx + syy
    return determinant - k * trace * trace


def select_local_maxima(
    response: np.ndarray, count: int = 250, nms_window: int = 5, relative_threshold: float = 0.01
) -> np.ndarray:
    """Trả về tọa độ (y,x) của các cực đại dương mạnh nhất."""
    maximum = float(response.max())
    if maximum <= 0:
        return np.empty((0, 2), dtype=int)
    local_max = cv2.dilate(response, np.ones((nms_window, nms_window), np.uint8))
    mask = (response == local_max) & (response >= relative_threshold * maximum)
    coordinates = np.argwhere(mask)
    order = np.argsort(response[mask])[::-1]
    return coordinates[order[:count]]


def mark_points(image: np.ndarray, points_yx: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    result = image.copy()
    for y, x in points_yx:
        cv2.circle(result, (int(x), int(y)), 3, color, 1, lineType=cv2.LINE_AA)
    return result


def exercise_harris(name: str, output_name: str, show: bool) -> None:
    image = read_color(name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    response = harris_response(gray, window=5, k=0.04)
    points = select_local_maxima(response, count=250, nms_window=5)
    manual = mark_points(image, points)

    cv_response = cv2.cornerHarris(np.float32(gray) / 255.0, blockSize=5, ksize=3, k=0.04)
    cv_mask = cv_response > 0.01 * cv_response.max()
    opencv = image.copy()
    opencv[cv_mask] = (0, 255, 0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    panels = [
        (image, "Gốc"),
        (manual, f"Harris tay + NMS ({len(points)} điểm)"),
        (opencv, "cv2.cornerHarris (ngưỡng 1%)"),
    ]
    for ax, (data, title) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / output_name, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Harris {name}: chọn {len(points)} cực đại địa phương")


def vote_orientation(histogram: np.ndarray, angle: float, magnitude: float) -> None:
    """Chia tuyến tính phiếu bầu cho hai bin cách nhau 45°, có vòng 315° -> 0°."""
    position = (angle % 360.0) / 45.0
    lower_unwrapped = int(np.floor(position))
    fraction = position - lower_unwrapped
    lower = lower_unwrapped % 8
    upper = (lower + 1) % 8
    histogram[lower] += (1.0 - fraction) * magnitude
    histogram[upper] += fraction * magnitude


def hog_cells(gray: np.ndarray, cell_size: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = gray.astype(np.float64)
    gx = cv2.Sobel(source, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(source, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 360.0) % 360.0
    cells_y, cells_x = gray.shape[0] // cell_size, gray.shape[1] // cell_size
    histograms = np.zeros((cells_y, cells_x, 8), dtype=np.float64)

    for cy in range(cells_y):
        for cx in range(cells_x):
            for y in range(cy * cell_size, (cy + 1) * cell_size):
                for x in range(cx * cell_size, (cx + 1) * cell_size):
                    vote_orientation(histograms[cy, cx], angle[y, x], magnitude[y, x])
            norm = np.linalg.norm(histograms[cy, cx])
            if norm > 0:
                histograms[cy, cx] /= norm
    return histograms, magnitude, angle


def exercise_hog(show: bool) -> None:
    image = read_color("C4_B2.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cell = 4
    histograms, magnitude, angle = hog_cells(gray, cell)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Ảnh xám")
    axes[1].imshow(magnitude, cmap="magma")
    axes[1].set_title("Độ lớn gradient")
    axes[2].imshow(gray, cmap="gray")
    axes[2].set_title("HOG từng cell 4×4")

    for cy in range(histograms.shape[0]):
        for cx in range(histograms.shape[1]):
            center_x = cx * cell + (cell - 1) / 2
            center_y = cy * cell + (cell - 1) / 2
            for bin_index, strength in enumerate(histograms[cy, cx]):
                if strength <= 0.02:
                    continue
                theta = np.deg2rad(bin_index * 45.0)
                half = 0.45 * cell * strength
                dx, dy = half * np.cos(theta), half * np.sin(theta)
                axes[2].plot(
                    [center_x - dx, center_x + dx],
                    [center_y - dy, center_y + dy],
                    color="lime",
                    linewidth=0.5 + 1.5 * strength,
                )
    for ax in axes:
        ax.axis("off")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / "bai2_hog.png", dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    np.save(OUT / "bai2_hog_vectors.npy", histograms)
    print("HOG shape (cells_y, cells_x, bins):", histograms.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    exercise_harris("C4_B1.png", "bai1_harris.png", args.show)
    exercise_harris("C4_B1_a.png", "bai1a_harris.png", args.show)
    exercise_hog(args.show)
    print(f"Đã lưu kết quả Chương 4 vào: {OUT}")


if __name__ == "__main__":
    main()
