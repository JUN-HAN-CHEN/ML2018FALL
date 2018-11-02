# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 21:08:28 2018

@author: j2831
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
features = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2',
        'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH',
        'SO2', 'THC', 'WD_HR', 'WIND_DIR', 'WIND_SPEED', 'WS_HR']

train = pd.read_csv('train.csv, encoding="big5")
for i in range(len(features)):
    filter1 = train["測項"]==features[i]
    pm25_train = train[filter1]
    pm25_train.drop(['日期', '測站', '測項'], axis = 1, inplace=True)
    pm25_train[pm25_train == 'NR'] = 0.0
    a=np.array(pm25_train)
    a=a.flatten()
    su=0
    a = a.astype('float')
#    for k in range(len(a)):
#        a[i]=float(a[i])
#        su+=a[i]
#    mean = su/len(a)
    mean=a.mean()
    std=a.std()
    for j in range(len(a)):
        if (a[j] > mean+5*std):
            a[j] = mean
#        elif (a[j] < mean+100*std):
#            a[j] = mean
    plt.plot(a)
    plt.ylabel(features[i])
    plt.show()
    print(mean)
    print(std)