from keras import applications
from keras.layers import Input, Dense, Conv2D, Activation, MaxPooling2D
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
import scipy.misc

f1 = '/home/ubuntu/data/test_data.npy'
f2 = '/Users/Nic/Desktop/Mining/Cancer/preprocessed_test.npy'

data = np.load(f1)

data = data / 255.

#print(set(data))

print "data shape:", data.shape


model1 = load_model('scratch_model1.h5')
model2 = load_model('scratch_model2.h5')
model3 = load_model('scratch_model3.h5')
model4 = load_model('scratch_model4.h5')
model5 = load_model('scratch_model5.h5')


preds1 = model1.predict(data, batch_size=1, verbose=1)
preds2 = model2.predict(data, batch_size=1, verbose=1)
preds3 = model3.predict(data, batch_size=1, verbose=1)
preds4 = model4.predict(data, batch_size=1, verbose=1)
preds5 = model5.predict(data, batch_size=1, verbose=1)

print preds1.shape
preds = (preds1 + preds5)
preds = np.true_divide(preds, 5.)

print 'done pred'
print "preds shape: ", preds.shape

import numpy as np
import pandas as pd

cols = ['Type_1', 'Type_2', 'Type_3']

nums = []

for i in range(512):
    nums.append('{}.jpg'.format(i))

nums = pd.DataFrame(nums, columns=['image_name'])
prediction = pd.DataFrame(preds, columns=cols)

result = pd.concat([nums, prediction], axis=1)
result.to_csv('scratch_prediction_ensemble3.csv', index=False)
