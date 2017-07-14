
#data taken from the competition: https://www.kaggle.com/c/titanic

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math

#loading train data and dropping columns with names and tickets
df = pd.read_csv("train.csv")
df = df.drop(df.columns[[3, 8]], axis=1)

#converting column with genders to 0 1
s = "female"
df.Sex = df.Sex == s
df.Sex = df.Sex.astype(int)

#converting embarked types into integers 0, 1 and 2
mapping = [('C','0'), ('S','1'), ('Q','2')]
for k,v in mapping:
    df.Embarked = df.Embarked.replace(k,v)

data = np.array(df)

#converting cabin types into integers
c = len(df) 
b = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
d = len(b)

for i in range(c):
    a = data[i, 8] != data[i, 8] #checking for nan values
    if a == True:
        e = 0
        e = str(e)
        data[i, 8] = e
    else:
        for j in range(d):
            if b[j] in data[i, 8]:
                e = j + 1
                e = str(e)
                data[i, 8] = e
                
#saving result to a DataFrame                
df.Cabin = data[:,8]

def train(data, epochs, batch):
    
    a = data.shape[1]-1
    dataY = data[:,1]
    dataX = data[:,2:a]
    
    model = Sequential()
    model.add(Dense(30, input_dim = dataX.shape[1], activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(5, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='logcosh', optimizer='adam', metrics=['accuracy'])
    model.fit(dataX, dataY, validation_split=.33, epochs=epochs, batch_size=batch, verbose=0)
    
    scores = model.evaluate(dataX, dataY)
    print("\n%s: %.2f%%\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100, model.metrics_names[0], scores[0]*100))
    
    b = int(round(.67 * dataX.shape[0] - 1 , 0))
    
    C = data[0:b,2:a]
    D = dataY[0:b]
    
    scores2 = model.evaluate(C, D, verbose = 0)
    print("\ntrain accuracy: %.2f%% \ntrain loss: %.2f%%" % (scores2[1]*100, scores2[0]*100))
    
    #the rest 33% - test loss and acc
    
    E = data[b:,2:a]
    F = dataY[b:]
    
    scores3 = model.evaluate(E, F, verbose = 0)
    print("\ntest accuracy: %.2f%% \ntest loss: %.2f%%" % (scores3[1]*100, scores3[0]*100))

#dropping rows with nan values
df = df.dropna()
data = np.array(df)
data = data.astype(int)

train(data, 270, 27)
