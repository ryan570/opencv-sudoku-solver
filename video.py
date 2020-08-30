import cv2
import numpy as np
from imutils.perspective import order_points
import time

from utils import *
from model import *
from norvig import solve

VIDEO_SOURCE = 'https://192.168.1.134:8080/video'
# changing this will also require changing the min and max values in utils.find_digit
WARPED_SIZE = 500

cap = cv2.VideoCapture(VIDEO_SOURCE)
empty_grid = cv2.imread('grid.png')

if (cap.isOpened() == False):
    print("Error opening video stream or file")


def area(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return w * h


read_now = False
has_been_read = False
solved = False
while(cap.isOpened()):
    start_time = time.time()
    in_frame = False
    success, frame = cap.read()

    if success:
        processed = process(frame)

        contours, hierarchy = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            biggest = sorted(contours, key=area, reverse=True)[:3]

            for cnt in biggest:
                polygon = cv2.approxPolyDP(
                    cnt, .02 * cv2.arcLength(cnt, True), True)
                x, y, w, h = cv2.boundingRect(polygon)
                if w / h > 0.7 and w / h < 1.3 and len(polygon) == 4:
                    in_frame = True
                    break

            if in_frame:
                polygon = polygon.reshape(4, 2)
                mask = np.zeros(frame.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], (255, 255, 255))
                masked = cv2.bitwise_and(frame, mask)
                polygon = order_points(polygon)
                M = cv2.getPerspectiveTransform(np.float32(polygon), np.float32(
                    [(0, 0), (WARPED_SIZE, 0), (WARPED_SIZE, WARPED_SIZE), (0, WARPED_SIZE)]))
                warped = cv2.warpPerspective(masked, M, (500, 500))
                warped = warped[:WARPED_SIZE, :WARPED_SIZE]

                no_grid = remove_grid(warped)

                if read_now:
                    step = 500 // 9
                    puzzle = np.zeros((9, 9))
                    for i in range(9):
                        for j in range(9):
                            square = no_grid[round(
                                i*step):round((i+1)*step), j*step:(j+1)*step]
                            found, digit = find_digit(square)

                            if found:
                                digit = pad(digit)
                                puzzle[i][j] = predict(digit)
                    read_now = False
                    has_been_read = True

                if has_been_read:
                    copy = empty_grid.copy()
                    for (x, y), num in np.ndenumerate(puzzle):
                        cv2.putText(copy, str(int(num)), ((y)*step + 12,
                                                        (1+x) * step - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                    cv2.imshow("puzzle", copy)
                
                if solved and in_frame:
                    blank = np.zeros((*frame.shape[:2][::-1], 4), dtype=np.uint8)
                    for (x, y), num in np.ndenumerate(solution):
                        cv2.putText(blank, str(int(num)), ((y)*step + 12,
                                                        (1+x) * step - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255, 255), 3)

                    M = cv2.getPerspectiveTransform(np.float32(
                        [(0, 0), (WARPED_SIZE, 0), (WARPED_SIZE, WARPED_SIZE), (0, WARPED_SIZE)]), np.float32(polygon))
                    blank_warp = cv2.warpPerspective(blank, M, frame.shape[:2][::-1])
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    frame = cv2.add(frame, blank_warp)
                    print("solved")

                cv2.imshow("warp", warped)
                frame = cv2.resize(frame, (500, 700))
                cv2.imshow('frame', frame)
                
                key = cv2.waitKey(1)
                if key & 0xFF == ord('s'):
                    to_solve = [str(int(s)) for s in puzzle.flatten()]
                    solved = solve(to_solve)
                    try:
                        solution = [*solved.values()]
                        solution = np.reshape(solution, (9, 9))

                        solved = True
                    except:
                        pass

                elif key & 0xFF == ord('r'):
                    solved = False
                    read_now = True

            else:
                pass
        print("FPS: ", 1.0 / (time.time() - start_time))

    else:
        break