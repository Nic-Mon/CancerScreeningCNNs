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

f1 = '/home/ubuntu/data/test_data.npy'
f2 = '/Users/Nic/Desktop/Mining/Cancer/preprocessed_test.npy'

data = np.load(f1)

#from sklearn.utils import shuffle
#data = shuffle(data,random_state=2)
#train_data = [data,labels]

data = data / 255.

print(data[0].shape)
print(data[0][2][32][2])
print(data[3].max())
#print(set(data))
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# y_train = y_train[:,1:]
# y_test = y_test[:,1:]

print "data shape:", data.shape

model = load_model('scratch_model14.h5')
print model.summary()

preds = model.predict(data, batch_size=32, verbose=1)
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
result.to_csv('scratch_preds3.csv', index=False)
