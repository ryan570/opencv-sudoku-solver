import cv2
import numpy as np
from imutils.perspective import order_points

from model import *
from norvig import solve
from utils import *

WARPED_SIZE = 500


def area(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return w * h


def run(img, data_gen=False):
    processed = preprocess(img)

    contours, hierarchy = cv2.findContours(
        processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        biggest = sorted(contours, key=area, reverse=True)[:3]

        for cnt in biggest:
            polygon = cv2.approxPolyDP(
                cnt, .02 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(polygon)
            if w / h < 0.7 or w / h > 1.3 or len(polygon) != 4:
                continue
            polygon = polygon.reshape(4, 2)
            polygon = order_points(polygon)
            break

        warped = warp(img, polygon, ((0, 0), (WARPED_SIZE, 0), (
            WARPED_SIZE, WARPED_SIZE), (0, WARPED_SIZE)), (WARPED_SIZE, WARPED_SIZE))
        no_grid = remove_grid(warped)

        step = WARPED_SIZE // 9
        puzzle = np.zeros((9, 9))
        for i in range(9):
            for j in range(9):
                square = no_grid[round(
                    i*step):round((i+1)*step), j*step:(j+1)*step]
                found, digit = find_digit(square)

                if found:
                    digit = pad(digit)
                    puzzle[i][j] = predict(digit)

        to_solve = [str(int(s)) for s in puzzle.flatten()]
        solved = solve(to_solve)
        if type(solved) == bool:
            print("Puzzle was read incorrectly, skipping...")
            return
        solution = [*solved.values()]
        solution = np.reshape(solution, (9, 9))

        blank = np.zeros((*img.shape[:2][::-1], 4), dtype=np.uint8)
        for (x, y), num in np.ndenumerate(solution):
            cv2.putText(blank, str(int(num)), ((y)*step + 12,
                                               (1+x) * step - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255, 255), 3)

        overlay = warp(blank, ((0, 0), (WARPED_SIZE, 0), (WARPED_SIZE, WARPED_SIZE),
                               (0, WARPED_SIZE)), polygon, (img.shape[1], img.shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img = cv2.add(img, overlay)

        cv2.imshow("process", processed)
        cv2.imshow("warp", warped)
        cv2.imshow("no grid", no_grid)
        cv2.imshow("solved", img)
        cv2.waitKey(0)
