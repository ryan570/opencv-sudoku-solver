import cv2
import matplotlib.pyplot as plt
import numpy as np


def warp(img, src_coords, dest_coords, dest_size):
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.fillPoly(mask, np.array([src_coords], 'int32'), (255, 255, 255))
    masked = cv2.bitwise_and(img, mask)

    M = cv2.getPerspectiveTransform(
        np.float32(src_coords), np.float32(dest_coords))
    warped = cv2.warpPerspective(masked, M, dest_size)

    return warped


def pad(img, requiredW=32, requiredH=32):
    """Return digit centered inside image of the required dimensions"""
    width, height = img.shape[1], img.shape[0]

    if width >= 32 and height >= 32:
        return cv2.resize(img, (32, 32))
    if width >= 32:
        img = cv2.resize(img, (32, height))
        width = 32
    elif height >= 32:
        img = cv2.resize(img, (width, 32))
        height = 32

    dW, dH = requiredW - width, requiredH - height
    wLeft, hTop = dW // 2, dH // 2
    wRight, hBottom = dW // 2 + 1 if wLeft < dW / \
        2 else dW // 2, dH // 2 + 1 if hTop < dH / 2 else dH // 2

    img = np.pad(img, ((hTop, hBottom), (wLeft, wRight)),
                 'constant', constant_values=255)
    img = np.expand_dims(img, -1)

    return img


def find_digit(img, minimum=200, maximum=1000):
    """Attempt to find a digit inside image and return the image cropped to the digit"""
    found = False
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    low_threshold = 255/3
    high_threshold = 255
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def area(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return w * h

    if contours:
        biggest = sorted(contours, key=area, reverse=True)[0]
        x, y, w, h = cv2.boundingRect(biggest)
        area = w * h

        if area > minimum and area < maximum:
            # print(area)
            found = True
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            img = img[y:y+h, x:x+w]

    return found, img


def remove_grid(img):
    """Return sudoku board with grid removed and only numbers visible"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 15, -2)

    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    cols = horizontal.shape[1]
    horizontal_size = cols // 10

    horizontalStructure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    rows = vertical.shape[0]
    verticalsize = rows // 10

    verticalStructure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    grid = cv2.bitwise_or(horizontal, vertical)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    grid = cv2.dilate(grid, kernel)

    grid = np.invert(grid)
    grid = np.expand_dims(grid, -1)

    no_grid = cv2.bitwise_and(gray, grid)
    no_grid = np.invert(no_grid)

    mean, stdDev = cv2.meanStdDev(no_grid)
    mean, stdDev = round(mean[0][0]), round(stdDev[0][0])

    im_bw = cv2.threshold(no_grid, mean - 1.3*stdDev,
                          255, cv2.THRESH_BINARY)[1]

    return im_bw


def preprocess(img):
    """Preprocess original image"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    return edges
