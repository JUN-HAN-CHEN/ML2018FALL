import sys
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils #One-hot encoding
from keras.callbacks import ModelCheckpoint

def expand(x_train, y_train):

    fx_train = np.empty((x_train.shape[0]*12, 48, 48, 1))
    fy_train = np.empty(len(y_train)*12)

    for ind in range(0, len(x_train)):
        tmp = x_train[ind]
        i = 12*ind
        fx_train[i+0] = np.pad(tmp[0:42,0:42,:], ((3,3),(3,3),(0,0)), 'constant')
        fx_train[i+1] = np.pad(tmp[6:48,0:42,:], ((3,3),(3,3),(0,0)), 'constant')
        fx_train[i+2] = np.pad(tmp[6:48,6:48,:], ((3,3),(3,3),(0,0)), 'constant')
        fx_train[i+3] = np.pad(tmp[0:42,6:48,:], ((3,3),(3,3),(0,0)), 'constant')
        fx_train[i+4] = np.pad(tmp[3:45,3:45,:], ((3,3),(3,3),(0,0)), 'constant')
        for j in range(5):
            fx_train[i+5+j] = np.fliplr(fx_train[i+j])
        fx_train[i+10] = tmp
        fx_train[i+11] = np.fliplr(tmp)
        fy_train[i:i+12] = y_train[ind]

    return fx_train, fy_train

def load_data():
    number=4000
    train_df = pd.read_csv(sys.argv[1])
    x_train = np.array( [ list(map(float,train_df["feature"].iloc[i].split())) for i in range(len(train_df)) ] )
    x_train = x_train.reshape( x_train.shape[0], 48, 48, 1)
    x_train/=255
    y_train = np.array( train_df["label"] )
    x_train, y_train = expand(x_train, y_train)
    y_train = np_utils.to_categorical(y_train, 7)
    test_df = train_df.iloc[0:number]
    x_test = np.array( [ list(map(float, test_df["feature"].iloc[i].split())) for i in range(len(test_df)) ] )
    x_test = x_test.reshape( x_test.shape[0], 48, 48, 1)
    x_test/=255
    y_test = np.array( test_df["label"] )
    y_test = np_utils.to_categorical(y_test, 7)

	
    return (x_train, y_train, x_test, y_test)



(x_train, y_train, x_test, y_test) = load_data()

data_dim = (48, 48, 1)
nb_class = 7

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=data_dim))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

#model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))

#model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.5))
#model.add(Dense(2048))
#model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(nb_class))
model.add(Activation('softmax'))

model.summary()

cb=ModelCheckpoint('model_tmp', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

sgd = SGD(lr=0.005, decay=0.00001, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
train_history = model.fit(x_train, y_train, batch_size=150, epochs=50, validation_data=(x_test, y_test), callbacks=[cb])
#train_history = model.fit(x_train, y_train, batch_size=100, epochs=100, validation_split=0.2, callbacks=[cb])


score = model.evaluate(x_test, y_test)
print ('Train Acc:', score[1] )
model.save("model_tmp")
print(train_history)
#visualization
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')
from matplotlib import pyplot as plt
plt.plot(train_history.history['loss'])  
plt.plot(train_history.history['val_loss'])  
plt.title('Train History')  
plt.ylabel('loss')  
plt.xlabel('Epoch')  
plt.legend(['loss', 'val_loss'], loc='upper left')  
plt.show() 
plt.plot(train_history.history['acc'])  
plt.plot(train_history.history['val_acc'])  
plt.title('Train History')  
plt.ylabel('acc')  
plt.xlabel('Epoch')  
plt.legend(['acc', 'val_acc'], loc='upper left')  
plt.show()

