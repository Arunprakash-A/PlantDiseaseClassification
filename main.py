# import all the necessary packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
import pickle
import h5py
from tensorflow.python.keras.models import save_model
from tensorflow.keras.optimizers import Adam

#.....Training/Testing Phase ..............#
# Load data (ensure the type: original or background subtracted)

pickle_in = open(" path to data","rb") # train or test data
X = pickle.load(pickle_in)

pickle_in = open("path to label","rb")
y = pickle.load(pickle_in)

# Normalize the data to a range between 0 to 1
X = X/255.0

# Create a model
NAME = 'Segmented8_3x50%dropAfterMax_100pix_128,64,32_WZ=3_2dense_50%drop_batch256_e20'

model = Sequential()
model.add(Conv2D(128, (3, 3),input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate = 0.5))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate = 0.5))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate = 0.5))

model.add(Flatten())
model.add(tf.keras.layers.Dense(512,activation = tf.nn.relu))
model.add(Dropout(rate = 0.25))
model.add(tf.keras.layers.Dense(256,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(27,activation =tf.nn.softmax))

# set the optimizer, loss function, metrics to use
opt = Adam(lr = 0.00085)
model.compile(optimizer= 'adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# log the information about loss, accuracy..
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# run the model
model.fit(X, y, batch_size=256, epochs=20, validation_split=0.1,callbacks = [tensorboard])

# Save the trained model
model.save('Name to save the model')

# Load the model
#model.load('Path to load the model') # remove # for testing 
