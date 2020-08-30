import cv2
import os

from image import run

path = 'puzzles'
for filename in os.listdir(path):
    img = cv2.imread(os.path.join(path, filename))
    if img.shape[0] > 2000:
        img = cv2.resize(img, (500,700))
    run(img)
