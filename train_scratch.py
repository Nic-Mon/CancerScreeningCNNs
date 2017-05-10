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

f1 = '/home/ubuntu/data/out.npy'
f2 = '/home/ubuntu/data/label.npy'

data = np.load(f1)
labels = np.load(f2)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=5)

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

model = load_model('scratch_model15.h5')
print model.summary()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    horizontal_flip=True, vertical_flip=True, rotation_range=90, width_shift_range=.1, height_shift_range=.1)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
    horizontal_flip=True, vertical_flip=True)

train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=32)

validation_generator = test_datagen.flow(
    X_test, y_test,
    batch_size=32)


model.fit_generator(
    train_generator,
    steps_per_epoch= 40,
    epochs=100,
    validation_data=validation_generator,
    validation_steps= 10)

train_score = model.evaluate(X_train, y_train)
test_score = model.evaluate(X_test, y_test)
model.save('scratch_model15.h5')
with open("ScratchScores4.txt", "w") as text_file:
    text_file.write("Score: {}, {}".format(train_score, test_score))
