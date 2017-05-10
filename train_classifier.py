import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from keras.utils import np_utils
from keras.models import load_model
import numpy as np
import scipy.misc
import Image

f1 = '/home/ubuntu/data/out.npy'
f2 = '/home/ubuntu/data/label.npy'

data = np.load(f1)
labels = np.load(f2)

#from sklearn.utils import shuffle
#data,labels = shuffle(data,labels, random_state=2)
#train_data = [data,labels]


# Separate data into training and testing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=4)

X_train = X_train / 255.
X_test = X_test / 255.

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

y_train = y_train[:,1:]
y_test = y_test[:,1:]

print "X_train shape:", X_train.shape
print "X_test shape:", X_test.shape
print "y_train shape:", y_train.shape
print "y_test shape:", y_test.shape

np.save(open('/home/ubuntu/data/train_data.npy', 'w'), X_train)
np.save(open('/home/ubuntu/data/test_data.npy', 'w'), X_test)
np.save(open('/home/ubuntu/data/train_labels.npy', 'w'), y_train)
np.save(open('/home/ubuntu/data/test_labels.npy', 'w'), y_test)

# dimensions of our images.
img_width, img_height = 224,224

top_model_weights_path = 'bottleneck_fc_model.h5'

def save_bottlebeck_features():
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    print model.summary()
    bottleneck_features_train = model.predict(X_train)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    bottleneck_features_validation = model.predict(X_test)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)
    print(bottleneck_features_train.shape)

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = y_train

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = y_test

    model = Sequential()
    model.add(Flatten(input_shape=train_data[0].shape))
    #model.add(Dense(1000, activation='relu'))
    #model.add(Dropout(.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.85))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=optimizers.SGD(lr=4e-4, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print model.summary()
    model.fit(train_data, train_labels,
              epochs=100,
              batch_size=16,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model()
