import pandas as pd
import numpy as np

features = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2',
        'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH',
        'SO2', 'THC', 'WD_HR', 'WIND_DIR', 'WIND_SPEED', 'WS_HR']
def ReadTrainData(filename):
    raw_data = pd.read_csv(filename, encoding='big5').values
    data = raw_data[:, 3:] # 12 months, 20 days per month, 18 features per day. shape: (4320 , 24)
    data[data == 'NR'] = 0.0
    data = data.astype('float')

    X, Y = [], []
    for i in range(0, data.shape[0], 18*20):
        # i: start of each month
        days = np.vsplit(data[i:i+18*20], 20) # shape: 20 * (18, 24)
        concat = np.concatenate(days, axis=1) # shape: (18, 480)
        for j in range(0, concat.shape[1]-9):
            X.append(concat[:, j:j+9].flatten())
            Y.append([concat[9, j+9]])
   #18 types * 9 hrs
    return np.array(X), np.array(Y) #471 datas a month*12 months = 5652

def ReadTestData(filename):
    raw_data = pd.read_csv(filename, header=None, encoding='big5').values
    data = raw_data[:, 2:]
    data[data == 'NR'] = 0.0
    data = data.astype('float')

    obs = np.vsplit(data, data.shape[0]/18)
    X = []
    for i in obs:
        X.append(i.flatten())

    return np.array(X)
class Linear_Regression():
    def __init__(self):
        pass
    def parameter_init(self, dim):
        self.b = 0
        self.W = np.zeros((dim, 1))

    def feature_scaling(self, X, train=False):    
        if train:
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
        return (X - self.min) / (self.max - self.min)
        
    def predict(self, X): 
        return np.dot(X, self.W) + self.b
        
    def RMSELoss(self, X, Y):
        return np.sqrt(np.mean((Y - self.predict(X))** 2) )
        
    def recover(self,X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        return X* (self.max - self.min)+self.min
    def train(self, X, Y, valid_X, valid_Y, epochs=100000, lr=1000000000): 
        
        batch_size = X.shape[0]
        W_dim = X.shape[1]
        self.parameter_init(W_dim)
        #X = self.feature_scaling(X, train=True)

        lr_b = 0
        lr_W = np.zeros((W_dim, 1))

        h=[]
        for epoch in range(epochs):
        
            # mse loss
            grad_b = -np.sum(Y - self.predict(X))/ batch_size
            grad_W = -np.dot(X.T, (Y - self.predict(X))) / batch_size
            #for i in range(-1,-37,-1):
#            grad_W[126:] = 0
            # adagrad
            lr_b += grad_b ** 2
            lr_W += grad_W ** 2
            #delete'WD_HR', 'WIND_DIR', 'WIND_SPEED', 'WS_HR'
            self.W[126:]=0
            #update
            self.b = self.b - lr / np.sqrt(lr_b) * grad_b
            self.W = self.W - lr / np.sqrt(lr_W) * grad_W
#            self.b = self.b - lr_b* grad_b
#            self.W = self.W - lr_W* grad_W
            self.W[126:]=0
            #
            print(epoch)
            print(self.RMSELoss(X,Y))
            h.append(self.RMSELoss(X,Y))
        #X = self.recover(X)
        return h
X,Y = ReadTrainData('Train2.csv')
model = Linear_Regression()
h = model.train(X,Y,X,Y)
X_test = ReadTestData('Test.csv')
Y_test = pd.read_csv("sampleSubmission .csv", encoding="big5")
end = model.predict(X_test)
Y_test.value=end
Y_test.to_csv('ans.csv',index=False)

import matplotlib.pyplot as plt
plt.plot(h)
plt.show()
#mean =Y.mean()
#std = Y.std()
#for i in range(len(Y)):
#    if (Y[i] > mean+3*std):
#        Y[i]=mean
#    if (Y[i] < mean-3*std):
#        Y[i]=mean
#x = np.arange(0,5652,1)
#y= model.predict(X)
#plt.plot(x,y)
##plt.plot(Y)
#plt.show()