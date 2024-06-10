import os
import cv2
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog, messagebox
from matplotlib import pyplot as plt
from matplotlib.widgets import Button  # Thêm import

from feature import extract_features

def cosine_similarity(ft1, ft2):
    return - (ft1 * ft2).sum() / (np.linalg.norm(ft1) * np.linalg.norm(ft2))

def select_image():
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính của Tkinter
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

def process_and_display_image(test_images_path):
    # Đường dẫn tới thư mục chứa ảnh test
    test_folder = 'Data/Test'

    # Đọc file CSV chứa các đường dẫn tới ảnh huấn luyện và các đặc trưng đã lưu trữ
    df = pd.read_csv('data.csv')
    shape = (256, 256)

    # Đọc ảnh test và trích xuất các đặc trưng từ ảnh test
    test_img = cv2.imread(test_images_path)
    test_color_mean, test_color_std, test_hog, test_color_hist = extract_features(test_img, shape)

    # Khởi tạo danh sách để lưu độ tương đồng và tên ảnh
    dsts = []
    images = []

    # Duyệt qua từng hàng trong file CSV và tính toán độ tương đồng
    for idx, row in df.iterrows():
        color_mean = np.load(row['color_mean'])
        color_std = np.load(row['color_std'])
        hog_mean = np.load(row['hog'])
        color_hist = np.load(row['color_hist'])
        
        color_mean_dst = cosine_similarity(color_mean, test_color_mean)
        color_std_dst = cosine_similarity(color_std, test_color_std)
        hog_dst = cosine_similarity(hog_mean, test_hog)
        color_hist_dst = cosine_similarity(color_hist, test_color_hist)
        
        total_dst = 0.25 * color_mean_dst + 0.25 * color_std_dst + 0.25 * hog_dst + 0.25 * color_hist_dst
        dsts.append(total_dst)
        images.append(row['name'])

    # Chuyển đổi danh sách độ tương đồng và tên ảnh thành mảng numpy
    dsts = np.array(dsts)
    images = np.array(images)

    # Sắp xếp độ tương đồng theo thứ tự giảm dần và lấy 3 ảnh có độ tương đồng cao nhất
    sorted_indices = dsts.argsort()
    top_3_images = images[sorted_indices][:3]
    top_3_scores = dsts[sorted_indices][:3]

    # Xác định lớp của từng ảnh trong top 3 ảnh có độ tương đồng cao nhất
    classes = ["cow", "deer", "dog", "horse", "goat", "voi"]
    labels = []
    for image in top_3_images:
        for item in classes:
            if item in image:
                labels.append(item)
                break

    # Hiển thị 3 ảnh có độ tương đồng cao nhất cùng với nhãn dự đoán và tỉ lệ phần trăm giống nhau
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Hiển thị ảnh test
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img_rgb = cv2.resize(test_img_rgb, (shape[1], shape[0]))
    axes[0].imshow(test_img_rgb)
    axes[0].axis('off')
    axes[0].set_title('Test Image')

    # Hiển thị 3 ảnh có độ tương đồng cao nhất
    for i, (image_path, label, score) in enumerate(zip(top_3_images, labels, top_3_scores)):
        train_img = cv2.imread(image_path)
        train_img_rgb = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
        train_img_rgb = cv2.resize(train_img_rgb, (shape[1], shape[0]))
        axes[i + 1].imshow(train_img_rgb)
        axes[i + 1].axis('off')
        axes[i + 1].set_title(f'Similar {i + 1}: {label}  {score:.2%}')

    # Tạo nút thoát
    def close_window(event):
        plt.close()
        main()

    plt.subplots_adjust(bottom=0.2)
    exit_button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
    exit_button = Button(exit_button_ax, 'Thoát')
    exit_button.on_clicked(close_window)
    plt.show()

def main():
    while True:
        test_images_path = select_image()
        if test_images_path:
            process_and_display_image(test_images_path)
        else:
            messagebox.showinfo("Thông báo", "Không có ảnh nào được chọn.")

if __name__ == "__main__":
    main()
