# Image Square Reassembly

This project asks you to unscramble an input image, that has been divided into a grid of square tiles. 
The tiles will be shuffled and potentially rotated. The task will be to develop an ML model to re-arrange 
the tiles to their original state. 

To facilitate the task, the tile in the top left corner will not be shifted nor rotated. 

## Overview

Much of this is demonstrated in the Overview_of_Puzzles.ipynb notebook.

- Load an image
- Split the image into equal-sized square pieces
- Randomly shuffle the squares

Your task: 
- Train a model to predict which tile shares a boundary with another tile
- Use your model to reconstruct the original image from the model’s output 
- You cannot use any images besides the 3 provided for your training data

## Goal

- Develop a data strategy to train the model
- Design and train a model to  identify which tiles should be next to each other
- Use your model to create a method to recreate the original image from the shuffled tile
