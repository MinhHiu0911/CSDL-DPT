import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

import pandas as pd
from preprocess import extract_features, to_csv


def cosine_similarity(ft1, ft2):
    return - (ft1 * ft2).sum() / (np.linalg.norm(ft1) * np.linalg.norm(ft2))
