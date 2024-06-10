import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

import pandas as pd
from hog import extract_features, to_csv


def cosine_similarity(ft1, ft2):
    return - (ft1 * ft2).sum() / (np.linalg.norm(ft1) * np.linalg.norm(ft2))


train_folder = 'Data/Train'
test_folder = 'Data/Test'
test_images_path = 'cow1.jpg'
df = pd.read_csv('data.csv')
shape = (256, 256)

test_img = cv2.imread(os.path.join(test_folder, test_images_path))
test_color_mean, test_color_std, test_hog = extract_features(test_img, shape)

dsts = []
images = []

for idx, row in df.iterrows():
    color_mean = np.load(row[2])
    color_std = np.load(row[3])
    hog_mean = np.load(row[4])
    color_mean_dst = cosine_similarity(color_mean, test_color_mean)
    color_std_dst = cosine_similarity(color_std, test_color_std)
    hog_dst = cosine_similarity(hog_mean, test_hog)
    dsts.append(color_mean_dst + hog_dst + color_std_dst)

    images.append(row[1])

dsts = np.array(dsts)
images = np.array(images)

top_ten = images[dsts.argsort()][0]

# Biểu đồ
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = cv2.resize(test_img, (shape[1], shape[0]))
plt.imshow(test_img)
plt.axis('off')
plt.title(top_ten)
plt.show()


classes = ["cow", "deer", "dog", "horse", "sheep", "goat"]
for item in classes:
    if item in top_ten:
        print(f"The most relevance class is {item}")
