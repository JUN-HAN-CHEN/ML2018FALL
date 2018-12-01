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

testFile = sys.argv[1]
outputFile = sys.argv[2]

#prediction
#testFile = "test.csv"
outputFile = "ans.csv"
def load_data1():
    test_df = pd.read_csv(sys.argv[1])
    x_test = np.array( [ list(map(float, test_df["feature"][i].split())) for i in range(len(test_df)) ] )
    x_test/=255
    return x_test
x_test = load_data1()
x_test = x_test.reshape( x_test.shape[0], 48, 48, 1)

model = load_model('model_tmp')
prd_class = model.predict_classes(x_test)
#plot_model(model, to_file='model.png')
#Output
a=[]
for i in range(7178):
    a.append(str(i))
Id = pd.DataFrame(a,columns=["id"])
value = pd.DataFrame({"label":[0]*7178})
result = pd.concat([Id, value], axis=1)
result['label'] = prd_class
#result.to_csv('ans.csv', index=False, encoding='big5')


result.to_csv(sys.argv[2], index=False, encoding='big5')




