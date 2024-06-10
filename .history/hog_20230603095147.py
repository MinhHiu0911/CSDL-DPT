import cv2
from skimage.feature import hog
import numpy as np


def compute_gradient(image):
    # Tính toán gradient theo trục x và trục y của ảnh
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    # Tính toán magnitude và góc của gradient
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    angle = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

    return magnitude, angle


def compute_histogram(magnitude, angle, cells_per_block):
    # Số bin trong histogram
    bins = 9

    # Kích thước ảnh
    height, width = magnitude.shape

    # Số lượng cell theo chiều ngang và chiều dọc
    cells_x = width // cells_per_block
    cells_y = height // cells_per_block

    # Tạo histogram rỗng
    histogram = np.zeros((cells_y, cells_x, bins))

    # Chia magnitude và angle thành các cell
    magnitude_cells = np.split(magnitude[:cells_y * cells_per_block, :cells_x * cells_per_block],
                               cells_y, axis=0)
    angle_cells = np.split(angle[:cells_y * cells_per_block, :cells_x * cells_per_block],
                           cells_y, axis=0)

    # Lặp qua từng cell
    for i in range(cells_y):
        for j in range(cells_x):
            # Lấy magnitude và angle của cell
            cell_magnitude = magnitude_cells[i][j]
            cell_angle = angle_cells[i][j]

            # Chia cell_angle thành các bin
            bin_values, _ = np.histogram(
                cell_angle, bins=bins, range=(0, 180), weights=cell_magnitude)

            # Lưu giá trị bin vào histogram
            histogram[i, j, :] = bin_values

    return histogram


def compute_hog(image, cells_per_block=16):
    # Chuyển đổi ảnh sang dạng grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tính toán gradient và magnitude
    magnitude, angle = compute_gradient(gray)

    # Tính toán histogram của các cell
    histogram = compute_histogram(magnitude, angle, cells_per_block)

    # Chuẩn hóa histogram
    histogram /= np.sum(histogram)

    # Tạo đặc trưng HOG bằng cách ghép các histogram của các cell
    hog_features = histogram.flatten()

    return hog_features


def compare_hog(hog_1, hog_2):
    return abs(hog_1 - hog_2).sum()
