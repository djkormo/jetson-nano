import sys
from keras.callbacks import EarlyStopping ,ReduceLROnPlateau,ModelCheckpoint
import matplotlib.pyplot as plt

def load_input_data(fraction=0.10,dataset='mnist'):
  ''' Function load_input_data (fraction=0.10,dataset='mnist')
      Loads  dataset from keras.datasets
      Default dataset is Mnist
      Additional data set is split by fraction into train/test datasets
   '''
  from keras.datasets import mnist, cifar10, fashion_mnist
  
  # load train and test data 
  # mnist dataset 
  if (dataset=='mnist'):
    ((X_train, Y_train), (X_test, Y_test)) = mnist.load_data()
  # zalando fashion dataset  
  if (dataset=='fashion'):
    ((X_train, Y_train), (X_test, Y_test)) = fashion_mnist.load_data()
  # cifar10 dataset  
  if (dataset=='cifar10'):   
    ((X_train, Y_train), (X_test, Y_test)) = cifar10.load_data()  
  val_perc = fraction
  val_count = int(val_perc * (X_train.shape[0]))
  
  # number of validate images
  #print(val_count)

  # first pick validation set from train_data/labels
  X_validate = X_train[:val_count,:]
  Y_validate = Y_train[:val_count,]

  # leave rest in training set
  X_train = X_train[val_count:,:]
  Y_train = Y_train[val_count:,]
  
  return X_train,Y_train,X_validate,Y_validate,X_test,Y_test


def show_input_data(X_train,Y_train,X_validate,Y_validate,X_test,Y_test):
  '''
    Function show_input_data  shows
    train (X_train,Y_train)
    validate (X_validate,Y_validate)
    test (X_test,Y_test)
    datasets shape
  '''
  
  print("X_train.shape: ",X_train.shape)
  print("Y_train.shape: ",Y_train.shape)
  print("X_validate.shape: ",X_validate.shape)
  print("Y_validate.shape: ",Y_validate.shape)
  print("X_test.shape: ",X_test.shape)
  print("Y_test.shape: " ,Y_test.shape)
        
    
def plot_input_data(X,Y,columns=5,rows=5,size=28,figsize=5):
  '''
   function  plot_input_data(X,Y,columns=5,rows=5,size=28,figsize=5)
   X - train data
   Y - train labels
   columns and rows for display
   size  size of images
   figsize  size of display
  '''
  import matplotlib.pyplot as plt
  import numpy as np
  # preview the images first
  plt.figure(figsize=(figsize,figsize))
  x, y = columns, rows
  for i in range(columns*rows):  
    plt.subplot(y, x, i+1) 
    _=plt.axis("off")
    _=plt.text(0, 0, np.argmax(Y[i], axis=-1), fontsize=14, color='blue')
    ## TODO correcting displaying images class
    plt.imshow(X[i].reshape((size,size)),interpolation='nearest',cmap=plt.cm.binary)
  plt.show()
        
def reshape_input_data(X_train,X_validate,X_test,Y_train,Y_validate,Y_test,size=28):
  '''
   function reshape_input_data(X_train,X_validate,X_test,Y_train,Y_validate,Y_test,size=28)
   Reshaping datasets into CNN  shape 
   reshaping and normalization
  '''  
  from keras.utils import np_utils
  # change input data to 
  X_train = X_train.reshape(X_train.shape[0], size, size,1).astype('float32') / 255
  X_validate=X_validate.reshape(X_validate.shape[0], size, size,1).astype('float32') / 255
  X_test = X_test.reshape(X_test.shape[0], size, size,1).astype('float32') / 255

  Y_train = np_utils.to_categorical(Y_train, 10)
  Y_validate = np_utils.to_categorical(Y_validate, 10)
  Y_test = np_utils.to_categorical(Y_test, 10)
  return X_train,X_validate,X_test,Y_train,Y_validate,Y_test        
        

def display_model(model,image_name='model.png'):
  '''
  function display_model(model,image_name='model.png')
  Displaying model as png file
  '''
  import matplotlib.pyplot as plt
  from tensorflow.keras.utils import plot_model
  from IPython.display import Image
  plot_model(model, to_file=image_name, show_shapes=True)

def compile_model(model): 
  '''
  function compile_model(model)
  Compiling model
  '''  
  model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy','mae'])


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
checkpointer = ModelCheckpoint('mnist_weights.{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True)


def fit_model(model,X_train, Y_train,X_validate, Y_validate,batch_size=32, epochs=10,verbose=1,callbacks=[]):
    '''
    function fit_model(model,X_train, Y_train,
    X_validate, Y_validate,batch_size=32,    epochs=10,verbose=1,callbacks=[])
    Training model on X_train data with Y_train labels 
    with validation data X_validate, Y_validate
    callbacks can be used
    '''
    model_history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
           verbose=verbose,
           # validation data                   
           validation_data=(X_validate, Y_validate),
           # Callbacks                 
           callbacks=callbacks)
    return model_history        
        
def model_evaluate(model,X,Y,verbose=1):
  '''
  function model_evaluate(model,X,Y,verbose=1)
  '''  
  scores=model.evaluate(X, Y,verbose=1)
  
  print(scores)
  print("%s: %.8f%%" % (model.metrics_names[0], scores[0]))
  print("%s: %.8f%%" % (model.metrics_names[1], scores[1]))
  print (model.metrics_names)
  return scores    

def plot_model_history(model_history,size=5):
  '''
  function plot_model_history(model_history,size=5)
  '''
  import matplotlib.pyplot as plt
  
  plt.figure(figsize=(size, size))
  acc = model_history.history['acc']
  val_acc = model_history.history['val_acc']
  loss = model_history.history['loss']
  val_loss = model_history.history['val_loss']

  epochs = range(1, len(acc) + 1)

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.legend()

  plt.figure(figsize=(size, size))

  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()

  plt.show()  
        
        
def model_prediction_summary(model,X,Y,verbose=1):
  '''
   function model_prediction_summary(model,X,Y,verbose=1,)
   arguments:
   
     model : trained model
     X: input data for prediction
     Y: labels for prediction
     return:
     predicted labels
  '''
  import pandas as pd
  import numpy as np
  Y_hat = model.predict_classes(X,verbose=1)
  print("Input shape: " ,X.shape)
  print("Label shape: " ,Y.shape)
  print("Predicted Label shape: " ,Y_hat.shape)
  print(pd.crosstab(Y_hat, np.argmax(Y, axis=-1)))
  
  return Y_hat       


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools 
    import matplotlib.pyplot as plt
    #from sklearn.metrics import confusion_matrix
    plt.figure(figsize=(8,8))
    _=plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
    
def plot_wrong_images(X,Y,Y_pred,size=10):

  import numpy as np
  test_wrong = [im for im in zip(X,Y_pred,np.argmax(Y, axis=-1)) if im[1] != im[2]]
  print("All items count: ",len(X))
  print("Wrong items count: ",len(test_wrong))
  print("Accuracy: ",(len(X)-len(test_wrong))/(len(X)) )

  _=plt.figure(figsize=(size, size))
  for ind, val in enumerate(test_wrong):
    _=plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _=plt.subplot(15, 15, ind + 1)
    im = 1 - val[0].reshape((28,28))
    _=plt.axis("off")
    _=plt.text(0, 0, val[2], fontsize=14, color='blue')
    _=plt.text(6, 0, '->', fontsize=14, color='black')
    _=plt.text(16, 0, val[1], fontsize=14, color='red')
    _=plt.imshow(im, cmap="gray")    
    
def model_save(model,schema_file='model.json',weights_file='model.h5'):

 #Save the model
 # serialize model to JSON
  model_json = model.to_json()
  with open(schema_file, "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights(weights_file)
  print("Saved model to disk")
  print("Model json to: ",schema_file," weights to: ",weights_file)    
    
# several cnn model for mnist dataset

def first_model(width=28,height=28,num_classes=10,verbose=0):
  '''
   function first_model(width=28,height=28,num_classes=10,verbose=0)
  '''

  from keras.models import Sequential
  from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu',
          input_shape=(width, height,1)),)
  
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Flatten())
  
  model.add(Dense(num_classes, activation='softmax'))
  
  return model

def second_model(width=28,height=28,num_classes=10,verbose=0):
  '''
  *
  '''  
  from keras.models import Sequential
  from keras.layers import Flatten,Conv2D, MaxPooling2D,Flatten,Dense
  from keras.layers import Dropout
  
  model = Sequential()
  
  model.add(Conv2D(32, (3, 3), activation='relu',
          input_shape=(width, height,1)))
  if (verbose>0):
    print(model.output_shape)
    
  model.add(Conv2D(32, (3, 3), activation='relu'))
  if (verbose>0):
    print(model.output_shape)
    
  model.add(MaxPooling2D(pool_size=(2, 2)))
  if (verbose>0):
    print(model.output_shape) 
    
  
  model.add(Dropout(0.25))
  if (verbose>0):
    print(model.output_shape) 
  
  
  model.add(Flatten())
  if (verbose>0):
    print(model.output_shape)

  model.add(Dense(num_classes, activation='softmax'))
  if (verbose>0):
    print(model.output_shape)
    
  return model

def third_model(width=28,height=28,num_classes=10,verbose=0):
  '''
  Third model
  '''  

  from keras.models import Sequential
  from keras.layers import Flatten,Conv2D, MaxPooling2D,Flatten,Dense
  from keras.layers import Dropout 
    
  model = Sequential()
  
  # add Convolutional layers
  model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                     input_shape=(width, height, 1)))
  
  model.add(MaxPooling2D(pool_size=(2,2)))
  
  model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
  
  model.add(MaxPooling2D(pool_size=(2,2)))
  
  model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
  
  model.add(MaxPooling2D(pool_size=(2,2)))    
  
  model.add(Flatten())
  
  # use Dropout
  model.add(Dropout(0.4))
  
  # Densely connected layers
  
  model.add(Dense(128, activation='relu'))
  
  # output layer
  model.add(Dense(num_classes, activation='softmax'))
  
  return model


def fourth_model(width=28,height=28,num_classes=10,verbose=0):
  '''
  fourth_model
  '''  
    
  from keras.models import Sequential
  from keras.layers import Flatten,Conv2D, MaxPooling2D,Flatten,Dense,BatchNormalization
  from keras.layers import Dropout    
  model = Sequential()
  
  model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (width, height, 1)))
  
  model.add(BatchNormalization())
  
  model.add(Conv2D(32, kernel_size = 3, activation='relu'))
  model.add(BatchNormalization())
  
  model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
  model.add(BatchNormalization())
  
  model.add(Dropout(0.4))

  model.add(Conv2D(64, kernel_size = 3, activation='relu'))
  model.add(BatchNormalization())
  
  model.add(Conv2D(64, kernel_size = 3, activation='relu'))
  model.add(BatchNormalization())
  
  model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
  model.add(BatchNormalization())
  
  model.add(Dropout(0.4))

  model.add(Conv2D(128, kernel_size = 4, activation='relu'))
  model.add(BatchNormalization())
  

  model.add(Flatten())
  
  model.add(Dropout(0.4))
  
  model.add(Dense(num_classes, activation='softmax'))
  
  return model    
    
def fifth_model(width=28,height=28,num_classes=10,verbose=0):  
  '''
  fifth_model(width=28,height=28,num_classes=10,verbose=0): 
  '''    
  from keras.models import Sequential
  from keras.layers import Flatten,Conv2D, MaxPooling2D,Flatten,Dense,BatchNormalization
  from keras.layers import Dropout    
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu')) 
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
  
  return model
