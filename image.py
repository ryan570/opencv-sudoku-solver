import cv2
import numpy as np
from imutils.perspective import order_points

from utils import *
from model import *
from norvig import solve

WARPED_SIZE = 500

def area(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return w * h

def run(img):
    processed = process(img)

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
            break

        polygon = polygon.reshape(4, 2)
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], (255, 255, 255))
        masked = cv2.bitwise_and(img, mask)
        polygon = order_points(polygon)
        M = cv2.getPerspectiveTransform(np.float32(polygon), np.float32([(0, 0), (500, 0), (500, 500), (0, 500)]))
        warped = cv2.warpPerspective(masked, M, (500, 500))
        warped = warped[:500, :500]

        no_grid = remove_grid(warped)
        step = 500 // 9
        puzzle = np.zeros((9,9))
        for i in range(9):
            for j in range(9):
                square = no_grid[round(i*step):round((i+1)*step), j*step:(j+1)*step]
                found, digit = find_digit(square)

                if found:
                    digit = pad(digit)
                    puzzle[i][j] = predict(digit)
        
        to_solve = [str(int(s)) for s in puzzle.flatten()]
        solved = solve(to_solve)
        if type(solved) ==  bool:
            return
        solution = [*solved.values()]
        solution = np.reshape(solution, (9, 9))

        blank = np.zeros((*img.shape[:2][::-1], 4), dtype=np.uint8)
        for (x, y), num in np.ndenumerate(solution):
            cv2.putText(blank, str(int(num)), ((y)*step + 12,
                                            (1+x) * step - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255, 255), 3)

        M = cv2.getPerspectiveTransform(np.float32(
            [(0, 0), (WARPED_SIZE, 0), (WARPED_SIZE, WARPED_SIZE), (0, WARPED_SIZE)]), np.float32(polygon))
        blank_warp = cv2.warpPerspective(blank, M, img.shape[:2][::-1])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img = cv2.add(img, blank_warp)

        cv2.imshow("solved", img)
        cv2.waitKey(0)