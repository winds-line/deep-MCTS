import cv2
import numpy as np


def process_img(image):
    image = cv2.resize(image, (80, 40))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_array = image[:, :, np.newaxis]
    return image_array
