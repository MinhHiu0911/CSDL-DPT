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
