#Import libraries
from tensorflow import keras
import tensorflow as tf
import pathlib
import numpy as np
import cv2
from PIL import ImageGrab, Image
import pyautogui
import pydirectinput
import time

# Load TensorFlow-model
model = keras.models.load_model('ad-model.h5')

# Set image size as it is expected by the model
img_height = 180
img_width = 180

# Get the different classes
data_dir = pathlib.Path("path/to/data") # Set path to folder with the classes/data
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir)
class_names = train_ds.class_names
print(class_names)

# Function for autonomous driving
def ad():
    while(True):
        pydirectinput.press('up') # Always press 'up'
        printscreen = np.array(ImageGrab.grab(bbox=(30, 265, 780, 525))) # Specify screen area to be processed by the CNN
        
        # Edit image and make prediction
        processed_img = cv2.cvtColor(printscreen, cv2.COLOR_RGB2GRAY)
        processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
        cv2.imshow('window', processed_img)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        processed_img = Image.fromarray(processed_img)
        processed_img = tf.image.resize(processed_img, [img_height, img_width])
        img_array = tf.keras.utils.img_to_array(processed_img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        direction = class_names[np.argmax(score)] # direction = class name
        print(direction)
        
        # Press keys depending on prediction
        if direction == 'sr':
            pydirectinput.keyDown('right')
            pydirectinput.keyDown('up')
            time.sleep(0.005)
            pydirectinput.keyUp('up')
            pydirectinput.keyUp('right')
        elif direction == 'hr':
            pydirectinput.keyDown('right')
            pydirectinput.keyDown('up')
            #time.sleep(0.005)
            pydirectinput.keyUp('up')
            time.sleep(0.35)
            pydirectinput.keyUp('right')
        elif direction == 'sl':
            pydirectinput.keyDown('left')
            pydirectinput.keyDown('up')
            time.sleep(0.005)
            pydirectinput.keyUp('up')
            pydirectinput.keyUp('left')
        elif direction == 'hl':
            pydirectinput.keyDown('left')
            pydirectinput.keyDown('up')
            pydirectinput.keyUp('up')
            time.sleep(0.05)
            pydirectinput.keyUp('left')
        else:
            pydirectinput.keyDown('up')
            time.sleep(0.03)
            pydirectinput.keyUp('up')

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

# Initiate autonomous driving-function
ad()
