# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 19:03:09 2018

@author: Alexander
"""

import numpy as np
import pandas as pd

#%%Задача 1
class OLS(object):
    def __init__(self, y, X):
        k = 3
        n = 100
        y.shape = (n, 1)
        X.shape = (n, k)
        self.y = y
        self.X = X
        XTX = (X.T @ X)
        XTX_1 = np.linalg.inv(XTX)
        beta = XTX_1 @ X.T @ y
        self.beta = beta
        sigma_2 = 1/(n-k) * (y - X @ beta).T @ (y - X @ beta)
        sigma = sigma_2**(1/2)
        self.sigma = sigma
        V_beta = sigma_2 * XTX_1
        self.V = V_beta
    def predict(self, x):
        yp = x.T @ self.beta
        Var_y = self.sigma * (1 + x.T @ np.linalg.inv(self.X.T @ self.X) @ x)     
        return (yp[0], Var_y[0][0])
    

X = np.random.randn(100,3)
y = X.dot(np.array([1,2,3]))+np.random.randn(100)
model = OLS(y, X)        


print(model.beta)
print(model.V)
print(model.predict(np.array([1,0,1])))

        
    
#%%Задача 2
#""" 1. Beta is comming"""
#
#k=list(range(1,11))
#for i in range(k):
#   def beta(k):
#       for i in range(k):
#            beta(k) = np.random.rand(200)
#""" 2. """
#X = np.random.uniform (-5, 5, 200)
#
#""" 3. """
#U = np.random.randn(200)
#
#""" 4. """
#k=list(range(1,11))
#for i in range(k):

# Первые четыре пункта второй задачи.
n=200
k=1
beta = np.random.rand(11)
x = np.random.uniform(-5,5,n)
u = 10*np.random.randn(200)

def fuck(n): #factorial (fucktorial)
    if n == 0:
        return 1
    return n*fuck(n-1)

y = np.zeros((200,1))

for i in range(200):
    y[i] = y[i] + u[i]
for j in range(11):
    y[i] = y[i] + beta[j]*(x[i]**j/fuck(j))

#Пятый поехал
plt.scatter(x,y)

#K=1
model_k1 = OLS(y,x)
print(model_k1.b)

yk1=[]
yk2=[]
yk3=[]
yk4=[]
for i in range(n):
    y_new = model_k1.predict(x[i])[0]
    yk1 = np.append(yk1,[y_new]) 

plt.plot(x,yk1, label = 'K=1')






#%%Задача 3


import scipy as sc
import numpy as np
import math
from scipy import stats
from scipy.stats import t
import matplotlib.pyplot as plt

#
#Для столбов - axis=0, для строк - axis=1
t = sc.stats.t.ppf(0.95, 99)
MatriXXX = np.random.randn(100, 100)
Stolb_mean = np.mean(MatriXXX, axis=0)
#Strok_mean = np.mean(MatriXXX, axis=1)
Stolb_std = np.std(MatriXXX, axis=0)
#Strok_std = np.std(MatriXXX, axis=1)
print (t)

#Boundaries
zero_left_stolb = Stolb_mean - t * Stolb_std / np.sqrt(len(MatriXXX))
zero_right_stolb = Stolb_mean + t * Stolb_std / np.sqrt(len(MatriXXX))

#Как делали в играх и Фибоначчи
x = []
count = 0
for i in range(len(zero_right_stolb)):
    if  zero_left_stolb[i] < 0 < zero_right_stolb[i]:
        x.append(True)
        count = count + 1
    else:
        x.append(False)
    
    
#Печатаем массив
for i in range (len(x)):
    print(x[i])
    

#zero_left_strok = Strok_mean - t * Strok_std / np.sqrt(len(MatriXXX))
#zero_right_strok = Strok_mean + t * Strok_std / np.sqrt(len(MatriXXX))
#np.mean(matrix, axis = 0/1)
#Mat[0]
#Mat.mean[axis=1]
#n = 100
#m = 100
#M = np.random.randn(1, 1)
#for i in range(n):
#    MatriXXX = [M] * n
#    
#    MatriXXX[i] = [M] * m








#%% Задача 4

import pandas as pd
#1.
dframe = pd.read_csv('goalies-2014-2016.csv', sep = ';')
dframe_1 = dframe.head(n=5)
dframe_1.iloc[:,0:6]

#2.

df = dframe.saves/dframe.shots_against
df1 = round(df,3)

avgdframe_1 = pd.Series.sum(df1)/len(df1)
absolutdev = df1-avgdframe_1

pd.Series.max(absolutdev)

#3.

pd.DataFrame.mean(df.games_played)
pd.DataFrame.mean(df.goals_against)
pd.DataFrame.mean(df.save_percentage)

pd.DataFrame.std(df.games_played)
pd.DataFrame.std(df.goals_against)
pd.DataFrame.std(df.save_percentage)

#4.

new_df = dframe[dframe['season']=='2016-17'][dframe['games_played'] > 40]
new_df = new_df[new_df.games_played > 40]
new_df= new_df.sort_values('save_percentage', ascending = False)
pd.DataFrame.max(new_df.save_percentage)
new_df.player[:1]

#5.
dframe_5 = []
dframe_5 = dframe.season
dfframe1617 = dframe[dframe.season == '2016-17']
dfframe1617 = dframe1617[['season','player', 'saves' ]]
dframe1516 = dframe[dframe.season == '2015-16']
dframe1516 = dframe1516[['season','player','saves']]
dframe1415 = dframe[dframe.season == '2014-15']
dframe1415 = dframe1415[['season','player','saves']]

dframe1617.loc[df['saves'] == pd.DataFrame.max(df1617.saves)]
dframe1516.loc[df['saves'] == pd.DataFrame.max(df1516.saves)]
dframe1415.loc[df['saves'] == pd.DataFrame.max(df1415.saves)]

#6.

keeper_date = df[df.wins >=30]
keeper_date = keeper_date[['season', 'player', 'wins']]

keeper_date.groupby('player').player.count()



