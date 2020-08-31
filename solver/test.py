from pathlib import Path

import cv2

from image import run

path = Path(__file__).parents[1] / 'puzzles/'

for filename in path.iterdir():
    img = cv2.imread(str(filename.resolve()))
    if img.shape[0] > 2000:
        img = cv2.resize(img, (500,700))
    run(img)
