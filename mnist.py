import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
import time

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

#image_index = 7777
#print(y_train[image_index])
#plt.imshow(x_train[image_index],cmap='Greys')
#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

input_shape = (28,28,1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

#creating a sequential model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
start_time = time.time()
history = model.fit(x=x_train,y=y_train,epochs=1,verbose=1,validation_split=0.25)
end_time = time.time()
processing_time = end_time-start_time
print("Time took for training : ",processing_time)

model.save('mnistTest.h5')

model.evaluate(x_test,y_test,verbose=1)

#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
#
## Plot training & validation loss values
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
#








