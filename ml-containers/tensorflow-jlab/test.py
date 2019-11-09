from tensorflow.python.client import device_lib
device_lib.list_local_devices()


from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

#loading data
print("Loading data...")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# creating model
print("Creating data...")
network = models.Sequential()
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# compiling model
print("Compiling data...")
network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# preparing input data
print("Preparing input data...")
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# training model
print("Traininig model ...")
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Evaluating model 
print("Evaluating model...")]
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc, 'test_loss', test_loss)













