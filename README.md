# OpenCV Sudoku Sover

## Overview
This project will read images or videos of Sudoku puzzles and display the solution back on the source. 

## Usage 
1. Install requirements with ``pip install -r requirements.txt``
2. Enter path to digit recognition model in ``solver/model.py``
3. Enter path to video in `solver/model.py`
4. Run ``solver/model.py``

Pressing F will display a frozen version of the solution and Q will quit.

## Methodology
This project uses OpenCV to read and process input video, Tensorflow to classify the digits in the puzzle, and [Peter Norvig's sudoku solver](https://norvig.com/sudopy.shtml) to solve the puzzle. 

An example of the stages each frame goes through can be seen below. 

1. Original puzzle 

![example puzzle](https://github.com/ryan570/opencv-sudoku-solver/blob/master/examples/puzzle.png?raw=true)

2. Preprocessed image

![example puzzle](https://github.com/ryan570/opencv-sudoku-solver/blob/master/examples/preprocessed.png?raw=true)

3. Warped image with digits only
   
![example puzzle](https://github.com/ryan570/opencv-sudoku-solver/blob/master/examples/warped_puzzle.png?raw=true)

4. Binary image with only digits
   
![example puzzle](https://github.com/ryan570/opencv-sudoku-solver/blob/master/examples/digits_only.png?raw=true)

5. Solution displayed on original frame
   
![example puzzle](https://github.com/ryan570/opencv-sudoku-solver/blob/master/examples/solution.png?raw=true)