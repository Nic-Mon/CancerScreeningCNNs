from keras import applications
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
import scipy.misc

# path to the model weights files.
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 224,224

train_data = np.load(open('/home/ubuntu/data/train_data.npy'))
train_labels = np.load(open('/home/ubuntu/data/train_labels.npy'))
validation_data = np.load(open('/home/ubuntu/data/test_data.npy'))
validation_labels = np.load(open('/home/ubuntu/data/test_labels.npy'))

# build the VGG16 network
input = Input(shape=(224,224,3))
base_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=input)
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=(7,7,512)))
top_model.add(Dense(1000, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(3, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

model = Model(inputs=base_model.input, outputs= top_model(base_model.output))

# add the model on top of the convolutional base

for layer in model.layers[:15]:
    layer.trainable = False

print(model.summary())

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=2, batch_size=64,validation_data=(validation_data, validation_labels))
model.save('bottleneck_model4.h5')
