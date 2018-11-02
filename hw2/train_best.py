import pandas as pd
import numpy as np
import sys
import time
start = time.time()
#wait to do
# correlation
# ADAM
#col_pay = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
#df[col_pay] = df[col_pay].apply(lambda x: x+2)
class Logistic_Regression():
    def __init__(self):
        pass
    def parameter_init(self, dim):
        self.b = 0
        self.W = np.zeros((dim, 1))

    def feature_scaling(self, X, train=False):    
        if train:
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
        return (X - self.min) / (self.max - self.min+10**-20)
        
    def predict(self, X,W,b): 
        self.W =W
        self.b = b
        return sigmoid(np.dot(X, self.W) + self.b)
        
    def RMSELoss(self, X, Y):
        return np.sqrt(np.mean((Y - self.predict(X))** 2) )
        
    def do_predict(self, X,W,b):
        threshold = 0.43
        X = self.feature_scaling(X, train=True)
        Y_pred = self.predict(X,W,b)
        Y_pred[Y_pred<threshold] = 0
        Y_pred[Y_pred >= threshold] = 1
        return Y_pred
    def train(self, X, Y, valid_X, valid_Y, epochs=100000, lr=0.01): 
        
        batch_size = X.shape[0]
        self.parameter_init(X.shape[1])
        X = self.feature_scaling(X, train=True)
        X=X
        lr_b = 0
        lr_W = np.zeros((X.shape[1], 1))
        #accc=[]
        losss=[]
        for epoch in range(epochs):
            # mse loss
            grad_b = -np.sum(Y - self.predict(X))
            grad_W =  -np.dot(X.T, (Y - self.predict(X)))

            # adagrad
            lr_b += grad_b ** 2
            lr_W += grad_W ** 2 

            #update
            self.b = self.b - lr / ( np.sqrt(lr_b) + 10**-20) * grad_b
            self.W = self.W - lr / ( np.sqrt(lr_W) + 10**-20) * grad_W
            #calculating loss = cross entropy
            loss = -np.mean(Y*np.log(self.predict(X))+(1-Y)*np.log(1-self.predict(X)))           
            losss.append(loss)
            print('epoch:{}\n Loss:{}\n'.format(epoch+1, loss))
        return self.do_predict(X), self.W, self.b, losss
def OneHotEncoding(df):
    a = pd.get_dummies(df['SEX'],prefix='SEX')
    b = pd.get_dummies(df['EDUCATION'],prefix='EDUCATION')
    c = pd.get_dummies(df['MARRIAGE'],prefix='MARRIAGE')
    d = pd.get_dummies(df['PAY_0'],prefix='PAY_0')
    i = pd.get_dummies(df['PAY_6'],prefix='PAY_6')
    df.drop(labels=["SEX","EDUCATION", "MARRIAGE", 'PAY_0','PAY_6'],axis="columns",inplace=True)
    df = pd.concat([df,a,b,c,d,i],axis=1)
    return df
def OneHotEncoding2(df):
    a = pd.get_dummies(df['SEX'],prefix='SEX')
    b = pd.get_dummies(df['EDUCATION'],prefix='EDUCATION')
    c = pd.get_dummies(df['MARRIAGE'],prefix='MARRIAGE')
    df.drop(labels=["SEX","EDUCATION", "MARRIAGE"],axis="columns",inplace=True)
    df = pd.concat([df,a,b,c],axis=1)
    return df
def sigmoid(z):
    return 1/(1+np.exp(-1*z))
W = np.load('model.npy')
b = np.load('modelb.npy') 
#test data
#test_X = pd.read_csv('test_x.csv', encoding='big5')
test_X = pd.read_csv(sys.argv[3], encoding='big5')

test_X = OneHotEncoding(test_X)
a=pd.DataFrame({"PAY_6_8":[0]*10000})
test_X = pd.concat([test_X, a],axis=1)
X_test = test_X.values
model = Logistic_Regression()
Y_test = model.do_predict(X_test,W,b)
Y_test = Y_test.astype(int)
#Output
a=[]
for i in range(10000):
    a.append("id_"+str(i))
Id = pd.DataFrame(a,columns=["id"])
value = pd.DataFrame({"Value":[0]*10000})
result = pd.concat([Id, value], axis=1)
result['Value'] = Y_test
#result.to_csv('ans.csv', index=False, encoding='big5')
result.to_csv(sys.argv[4], index=False, encoding='big5')
