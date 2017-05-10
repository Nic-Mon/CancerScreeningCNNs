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
import Image

f1 = '/home/ubuntu/data/out.npy'
f2 = '/home/ubuntu/data/label.npy'

data = np.load(f1)
labels = np.load(f2)

num_samples = np.size(data) # 100

#from sklearn.utils import shuffle
#data,labels = shuffle(data,labels, random_state=2)
#train_data = [data,labels]

# Separate data into images and labels
#(X, y) = (train_data[0],train_data[1])

# Separate data into training and testing
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Resize X
# X_train = X_train.reshape(X_train.shape[0],3, img_rows, img_cols)
# X_test = X_test.reshape(X_test.shape[0],3, img_rows, img_cols)

# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')

#X_train /= 255
#X_test /= 255
data = data / 255.

labels = np_utils.to_categorical(labels)
labels = labels[:,1:]

#y_train = np_utils.to_categorical(y_trains)
#y_test = np_utils.to_categorical(y_test)

#y_train = y_train[:,1:]
#y_test = y_test[:,1:]

print 'Data shape:', data.shape
print 'Labels shape:', labels.shape

#print "X_train shape:", X_train.shape
#print "X_test shape:", X_test.shape
#print "y_train shape:", y_train.shape
#print "y_test shape:", y_test.shape

# path to the model weights files.

# weights_path = '../keras/examples/vgg16_weights.h5'
# top_model_weights_path = 'fc_model.h5'

# dimensions of our images.
img_width, img_height = 224,224

# train_data_dir = 'cats_and_dogs_small/train'
# validation_data_dir = 'cats_and_dogs_small/validation'

# build the VGG16 network print('Model loaded.')

input = Input(shape=[224,224,3])
vgg16 = applications.VGG16(weights="imagenet", include_top=False)
x = vgg16(input)
x = Flatten()(x)
#x = Dense(1000, activation='relu')(x)
#x = Dropout(.1)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(.2)(x)
preds = Dense(3, activation='softmax')(x)

model = Model(inputs=input, outputs=preds)

# build a classifier model to put on top of the convolutional model
#top_model = Sequential()
#top_model.add(Flatten(input_shape=model.output_shape[1:]))
#top_model.add(Dense(256, activation='relu'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning

#top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
#count = 0
#for layer in model.layers[:2]:
#    layer.trainable = False

for layer in vgg16.layers[:-4]:
    layer.trainable = False

print model.summary()

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=2e-4, momentum=0.9),
              metrics=['accuracy'])

#init_score = model.evaluate(X_test, y_test, batch_size=32)
#acc = model.accuracy(X_test, y_test, batch_size=32)
#with open("Initialscore3.txt", "w") as text_file:
#    text_file.write("Score: {}".format(init_score))
#    text_file.write("Accuracy: {}".format(acc))

model.fit(data, labels, batch_size=32, epochs=50, verbose=1)

#score = model.evaluate(data, labels, batch_size=32)
#acc = model.accuracy(X_test, y_test, batch_size=32)
#with open("score5.txt", "w") as text_file:
#    text_file.write("Score: {}".format(score))
#    text_file.write("Accuracy: {}".format(acc))

model.save('model5.h5')
