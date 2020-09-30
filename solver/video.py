import threading
import time
from pathlib import Path

import cv2
import numpy as np
from imutils.perspective import order_points

from model import *
from norvig import solve
from utils import *

WARPED_SIZE = 500
STEP = WARPED_SIZE // 9


def area(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return w * h


class Solver:
    def __init__(self, source):
        self.source = source
        self.solved = False
        self.to_display = []
        self.in_frame = False
        self.frames_since_seen = 0
        self.freeze_solution = False

    def run(self):
        cap = cv2.VideoCapture(self.source)
        empty_grid = None
    
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        while(cap.isOpened()):
            start_time = time.time()
            self.in_frame = False
            success, frame = cap.read()

            if success:
                processed = preprocess(frame)

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
                        self.in_frame = True
                        break

                cv2.putText(frame, str(self.frames_since_seen), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
                if self.in_frame:
                    self.frames_since_seen = self.frames_since_seen - 1 if self.frames_since_seen > 0 else 0  
                    if threading.active_count() < 2 and not self.solved:
                        threading.Thread(target=self.read_and_solve, args=(frame, polygon)).start()

                    if self.solved:
                        if len(self.to_display) == 0:
                            blank = np.zeros((*frame.shape[:2][::-1], 4), dtype=np.uint8)
                            for (x, y), num in np.ndenumerate(self.solution):
                                cv2.putText(blank, str(int(num)), ((y)*STEP + 12,
                                                                (1+x) * STEP - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255, 255), 3)
                            self.to_display = blank
                        
                        overlay = warp(self.to_display, ((0, 0), (WARPED_SIZE, 0), (WARPED_SIZE, WARPED_SIZE), (
                            0, WARPED_SIZE)), polygon, (frame.shape[1], frame.shape[0]))
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                        frame = cv2.add(frame, overlay)
                else:
                    self.frames_since_seen += 1

                    if self.frames_since_seen > 5:
                        self.frames_since_seen = 5
                        self.to_display = []
                        self.solved = False

                frame = cv2.resize(frame, (500, 700))
                cv2.imshow('frame', frame)

                if self.freeze_solution:
                    cv2.imshow('solution', empty_grid)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                elif key & 0xFF == ord('f'):
                    if self.solved:
                        empty_grid = cv2.imread(str(Path(__file__).parents[1].resolve() / 'grid.png'))
                        for (x, y), num in np.ndenumerate(self.solution):
                            cv2.putText(empty_grid, str(int(num)), ((y)*STEP + 12,
                                                            (1+x) * STEP - 10), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
                        self.freeze_solution = True
                    

                # print("FPS: ", 1.0 / (time.time() - start_time))

            else:
                break

    def read_and_solve(self, frame, polygon):
        warped = warp(frame, polygon, ((0, 0), (WARPED_SIZE, 0), (WARPED_SIZE,
                                                                  WARPED_SIZE), (0, WARPED_SIZE)), (WARPED_SIZE, WARPED_SIZE))
        no_grid = remove_grid(warped)

        puzzle = np.zeros((9, 9))
        for i in range(9):
            for j in range(9):
                square = no_grid[round(
                    i*STEP):round((i+1)*STEP), j*STEP:(j+1)*STEP]
                found, digit = find_digit(square)

                if found:
                    digit = pad(digit)
                    puzzle[i][j] = predict(digit)

        to_solve = [str(int(s)) for s in puzzle.flatten()]
        solved = solve(to_solve)
        try:
            solution = [*solved.values()]
            solution = np.reshape(solution, (9, 9))

            self.solution = solution
            self.solved = True
        except:
            pass


if __name__ == "__main__":
    solver = Solver('https://192.168.1.134:8080/video')
    solver.run()
