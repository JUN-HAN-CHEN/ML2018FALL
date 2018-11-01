import pandas as pd
import numpy as np
import sys
def sigmoid(z):
    return 1/(1+np.exp(-1*z)+10**-20)
def doPredict(xtest,W,b):
    probability = sigmoid(np.matrix(xtest)*W.T+b)
    threshold = 0.45
    return [1 if p >= threshold else 0 for p in probability]
def feature_scaling(X,train = False):    
    if train:
        min1 = np.min(X, axis=0)
        max1 = np.max(X, axis=0)
    return (X - min1) / (max1 - min1+10**-20)
def normalization(df):
    return (df-df.min()) / (df.max()-df.min() )
#data processing
#xtest = pd.read_csv("test_x.csv", encoding='big5')
xtest = pd.read_csv(sys.argv[3], encoding='big5')

xtest = normalization(xtest)

#train
W = np.load('Gmodel.npy')
b = np.load('Gmodelb.npy') 
y = np.matrix(xtest)*W.T+b
probability = sigmoid(y)
threshold = 0.43
ans = probability
for i in range(10000):
    if ( ans[i,0] < threshold):
        ans[i,0] =0
    else:
        ans[i,0] = 1
ans = ans.astype(int)
#Output
a=[]
for i in range(10000):
    a.append("id_"+str(i))
Id = pd.DataFrame(a,columns=["id"])
value = pd.DataFrame({"Value":[0]*10000})
result = pd.concat([Id, value], axis=1)
result['Value'] = ans
#result.to_csv('ans.csv', index=False, encoding='big5')
result.to_csv(sys.argv[4], index=False, encoding='big5')
