# autonomous-driving-trackmania

This Python program uses a previously trained Convolutional Neural Network (CNN) to predict if the car has to turn, in which direction it has to turn as well as how much steering has to be done. To do this, the CNN continuously processes the images from the game and classifies them (hard right turn, slight right turn, etc...). Based on the classification, the corresponding keyboard inputs are being sent to the game by the Python program. The CNN has been trained with labelled images from the game.

An example for a self-driving car in Trackmania for which this program has been used can be seen here:
https://www.youtube.com/watch?v=BTC8QZOIFgc
