import pandas as pd
import numpy as np
import sys
def sigmoid(z):
    return 1/(1+np.exp(-1*z)+10**-20)
def doPredict(xtest):
    W = (mean1-mean0).T*inverse_covariance
    b = (.5*mean1.T*inverse_covariance*mean1+.5*mean0.T*inverse_covariance*mean0)+np.log(len(x1df)/len(x0df))
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
#xdf = pd.read_csv('train_x.csv',encoding='big5')
#ydf = pd.read_csv('train_y.csv',encoding='big5')
#xtest = pd.read_csv("test_x.csv", encoding='big5')
xdf = pd.read_csv(sys.argv[1],encoding='big5')
ydf = pd.read_csv(sys.argv[2],encoding='big5')
xtest = pd.read_csv(sys.argv[3], encoding='big5')
xdf = normalization(xdf)
ydf = normalization(ydf)
xtest = normalization(xtest)
xdf["label"] = pd.Series(np.array(ydf).ravel())
x0df = xdf[xdf.label==0]
x1df = xdf[xdf.label==1]
x0df = x0df.drop(["label"], axis=1)
x1df = x1df.drop(["label"], axis=1)
xdf = xdf.drop(["label"], axis=1)

#Calculate
mean0 = np.matrix(x0df.mean()).T
mean1 = np.matrix(x1df.mean()).T
covariance= np.cov(np.matrix(xdf).T)
inverse_covariance = np.linalg.pinv(covariance)
#train
W = (mean1-mean0).T*inverse_covariance
b = (-0.5*mean1.T*inverse_covariance*mean1+.5*mean0.T*inverse_covariance*mean0)+np.log(len(x1df)/len(x0df))

np.save('Gmodel.npy',W)
np.save('Gmodelb.npy',b)

y = np.matrix(xtest)*W.T+b
y1 = np.matrix(xdf)*W.T+b
probability = sigmoid(y)
probability1 = sigmoid(y1)
threshold = 0.43
ans = probability
ans1 = probability1
for i in range(10000):
    if ( ans[i,0] < threshold):
        ans[i,0] =0
    else:
        ans[i,0] = 1
ans = ans.astype(int)
for i in range(10000):
    if ( ans1[i,0] < threshold):
        ans1[i,0] =0
    else:
        ans1[i,0] = 1
ans1 = ans1.astype(int)
print("Accuracy")
right=0
Y_train = ydf.values
for i in range(20000):
    if (Y_train[i,0] == ans1[i,0]):
        right+=1
print(right/20000)
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
 

