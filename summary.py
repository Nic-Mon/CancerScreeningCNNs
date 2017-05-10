from keras.layers import Input, Dense
from keras.models import Model
from keras import applications
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.models import load_model
import numpy as np
import pandas as pd
import scipy.misc
import Image

model2 = load_model('model2.h5')

print(model2.summary())

model3 = load_model('model3.h5')

print(model3.summary())