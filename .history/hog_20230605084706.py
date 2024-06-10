# importing required libraries
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def _hog(image, shape):
    block_size = 16
    cell_size = 8
    assert (image.shape[0] % cell_size == 0 and image.shape[1] %
            cell_size == 0), "Size not supported"
    nbins = 9
    dx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    dy = dx.T
    # tinh -1: độ sâu
    gx = cv2.filter2D(image, -1, dx)
    gy = cv2.filter2D(image, -1, dy)
    #
    gs = np.sqrt(np.square(gx) + np.square(gy))
    phis = np.arctan(gy / (gx + 1e-6))
    phis[gx == 0] = np.pi / 2

    argmax_g = gs.argmax(axis=-1)

    # lấy ra g, phi mà tại đó g max
    g = np.take_along_axis(gs, argmax_g[..., None], axis=1)[..., 0]
    phi = np.take_along_axis(phis, argmax_g[..., None], axis=1)[..., 0]
    histogram = np.zeros(
        (g.shape[0] // cell_size, g.shape[1] // cell_size, nbins))
    for i in range(0, g.shape[0] - cell_size + 1, cell_size):
        for j in range(0, g.shape[1] - cell_size + 1, cell_size):
            g_in_square = g[i:i + cell_size, j:j + cell_size]
            phi_in_square = phi[i:i + cell_size, j:j + cell_size]

            bins = np.zeros(9)

            for u in range(0, g_in_square.shape[0]):
                for v in range(0, g_in_square.shape[1]):
                    g_pixel = g_in_square[u, v]
                    phi_pixel = phi_in_square[u, v] * 180 / np.pi
                    bin_index = int(phi_pixel // 20)
                    a = bin_index * 20
                    b = (bin_index + 1) * 20

                    value_1 = (phi_pixel - a) / 20 * g_pixel
                    value_2 = (b - phi_pixel) / 20 * g_pixel

                    bins[bin_index] += value_2
                    bins[(bin_index + 1) % 9] += value_1

            histogram[int(i / cell_size), int(j / cell_size), :] = bins

    t = block_size // cell_size
    hist = []
    for i in range(0, histogram.shape[0] - t + 1):
        for j in range(0, histogram.shape[1] - t + 1):
            block = histogram[i:i + t, j:j + t, :]
            block = block.flatten()
            block /= np.linalg.norm(block) + 1e-6
            hist.append(block)

    hist = np.array(hist)

    return hist.flatten()


def _extract_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_cpy = cv2.threshold(
        gray, 60, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    image_cpy = cv2.bitwise_not(image_cpy)

    # Cut just the object out
    sums = image_cpy.sum(axis=0)
    t = np.where(sums != 0)
    x1, x2 = t[0][0], t[0][-1]
    sums = image_cpy.sum(axis=1)
    t = np.where(sums != 0)
    y1, y2 = t[0][0], t[0][-1]

    return image[y1:y2 + 1, x1:x2 + 1]


def _get_color_mean(image):
    # Lấy trung bình 3 từng kênh L*, a*, b*
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    obj_color_mean = np.array(lab.mean(axis=(0, 1))[:3])
    obj_color_mean = obj_color_mean.reshape((-1, 1))

    return obj_color_mean


def _get_color_std(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    obj_color_std = np.array(lab.std(axis=(0, 1))[:3])
    obj_color_std = obj_color_std.reshape((-1, 1))

    return obj_color_std


def _pad_resize(image, shape):
    image = cv2.resize(image, (shape[1], shape[0]))
    return image


def extract_features(image, shape, name=""):
    assert image.shape[-1] == 3, "Expected 3 channels, got %d" % image.shape[-1]
    image = _extract_object(image)

    obj_color_mean = _get_color_mean(image)
    obj_color_std = _get_color_std(image)
    # Resize về cùng một cỡ và đệm 1 vòng pixel 0 bên ngoài,
    # bỏ comment ở dưới sẽ thấy
    image = _pad_resize(image, shape)
    # Chồng hog và màu thành 1 vector

    feature = obj_color_mean, obj_color_std, _hog(image, shape)

    return feature


def to_csv():
    path = "Data"
    FJoin = os.path.join
    files = [FJoin(path, f) for f in os.listdir(path)]
    # print(files)
    # train_folder = "Data"
    shape = (256, 256)
    data = []
    labels = []

    for path in files[:1]:
        all_images = [FJoin(path, f) for f in os.listdir(path)]
        # print(all_images)
        for image in all_images[:1]:
            img = cv2.imread(image)
            color_mean, color_std, hog = extract_features(img, shape, image)
            print(image)
            labels.append(path)

            # color_std_output_path = os.path.join("Feature", 'color_out_' + name.split('.')[0] + '.csv')
            # color_std_output_path = open(color_std_output_path, "w")
            # color_std = [str(f[0]) for f in color_std]
            # color_std_output_path.write("%s\n" % (",".join(color_std)))
            # color_std_output_path.close()
    # print(labels)


to_csv()
