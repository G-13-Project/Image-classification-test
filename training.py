from keras.layers.pooling import MaxPooling2D
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pickle

pickle_in = open(
    r"E:\NDT\Academic\Sem_3\Project I\Testing\test_1\x.pickle", "rb")
x = pickle.load(pickle_in)

pickle_in = open(
    r"E:\NDT\Academic\Sem_3\Project I\Testing\test_1\y.pickle", "rb")
y = pickle.load(pickle_in)


# Scale down the picels (Max pices size is 255)
# normaization
x = x/255.0


# create models
model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=x.shape[1:]))
model.add(Activation('relu'))  # relu activation function
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # like nurals

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1))  # like brain

model.add(Activation('sigmoid'))

# adam is optimizing algorithm
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# train the model
# baatch size is how much img train in one time..... validation is 30% for validation
model.fit(x, y, batch_size=4, epochs=10, validation_split=0.3)

model.save(r"E:\NDT\Academic\Sem_3\Project I\Testing\test_1\Dogs_vs_Cats_CNN.model")
