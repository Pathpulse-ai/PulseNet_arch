# Import required libraries. TF version used is 2.9.2

%matplotlib inline

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch
import cv2
import random


from PIL import Image, ImageDraw
from PIL import ImageFont
from IPython.display import Image as displayImage

font = ImageFont.truetype("Type Machine.ttf", 10)

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout, Concatenate, Reshape
import keras.losses

# from tensorflow.keras.layers.core import Reshape

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

tf.config.run_functions_eagerly(True) 

print('Using TensorFlow version', tf.__version__)

np.set_printoptions(threshold=np.inf) # increase print threshold to infinity to view full numpy outputs
