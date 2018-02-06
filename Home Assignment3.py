#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:39:23 2018

@author: silis123
"""

#импортируем используемые библиотеки
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats

#%% 
#задание 1 создаем класс

X=np.random.randn(100,3)

y=X.dot(np.array([1,2,3]))+np.random.randn(100)

class OLS:
    def __init__(self, y, X):
        self.y = y
        self.X = X

    #пропишем необходимые функции
    def beta(self):
        invatrix = np.linalg.inv(np.dot(self.X.transpose(), self.X))
        beta = np.dot(np.dot((invatrix), self.X.transpose()), self.y)
        return beta
    
    def std(self):
        col, line = self.X.shape
        qk = (self.y - np.dot(self.X, self.beta())).transpose()
        pk = self.y - np.dot(self.X, self.beta())
        std = (1 / (col - line)) * np.dot(qk, pk)
        return std      
    
    def V(self):
        col, line = self.X.shape
        invatrix = np.linalg.inv(np.dot(self.X.transpose(), self.X))
        qk = (self.y - np.dot(self.X, self.beta())).transpose()    
        pk = self.y - np.dot(self.X, self.beta())
        std = (1 / (col - line)) * np.dot(qk, pk)
        V = std * invatrix
        return V
    
    def predict(self, array):
        predy = np.dot(array.transpose(), self.beta())
        col, line = self.X.shape
        invatrix = np.linalg.inv(np.dot(self.X.transpose(), self.X))
        qk = (self.y - np.dot(self.X, self.beta())).transpose()
        pk = self.y - np.dot(self.X, self.beta())
        std = (1 / (col - line)) * np.dot(qk, pk)
        Vy = 1 + (np.dot((np.dot(array.transpose(), invatrix)), array))
        Vy = std * Vy
        return predy, Vy
    
  
model=OLS(y,X)
print(model.beta())
print(model.V())
print(model.predict(np.array([1,0,1])))
#%% 
#Задание 2
#2.1
betaarand = np.random.rand(11)
#2.2
xarand = np.random.uniform(-5, 5, 200)
#2.3
uarand = 10 * np.random.randn(200)
#2.4
def factorial(N):
    if (N==0):
        return(1)
    return (N*factorial(N-1))
    

yarand = np.zeros((200,1))
for i in range(0, 200):
    sss = 0
    for j in range(0,11):
        sss = sss + betaarand[j] * (xarand[i]**j) / factorial(j)  
    yarand[i] = sss + uarand[i]
 
#2.5 график
xarand = xarand.reshape(200,1)  
plt.scatter(xarand, yarand, alpha = 0.6)

#Случай k = 1
reg1 = OLS(yarand, xarand)
y_hat1, V = reg1.predict(xarand.transpose())
plt.plot(xarand, y_hat1, label='K=1')
plt.show() 

# для случая k=2...4 не хватает времени, но я так понимаю, что в предикт надо просто вставить новый аргумент x**2...x**4, надеюсь, что эта интуиция чего-то стоит
#%%

#%%Задание 3
A = np.random.normal(size=(100, 100))

# fПеременные функции ex3_solver:
#   A - матрица
#   оси 0 или 1
# выводит на печать логические переменные True и их количество
def ex3_solver(A, axis):
    N = np.shape(A)[axis]
    mean = A.mean(axis = axis)

    # список несмещённых standard deviations колонок (ddof=1, следовательно дивизор N-1, следовательно, нет смещения)
    std = A.std(axis=axis, ddof=1)

    #возьмем квантиль из Student distribution
    quantiles =scipy.stats.t.ppf(0.9, N-1)

    # расчет границ средних
    right_bound = mean + std * quantiles / np.sqrt(N)
    left_bound = mean - std * quantiles / np.sqrt(N)

    does_belong = list(map(lambda x,y: True if (x>0 and y<0) else False, right_bound, left_bound))

    # кол-во листов (колонны или строки, в зависимости от оси) для которы "True" принадлежит к соответствующему confidence interval
    number_of_trues = does_belong.count(True)

    if axis == 0:
        print(does_belong, "\nAmount of \"True\" for columns:", number_of_trues, "\n")
    if axis == 1:
        print(does_belong, "\nAmount of \"False\" for rows:", number_of_trues, "\n")

# столбцы
ex3_solver(A, axis=0)

# строки
ex3_solver(A, axis=1)
#%%
#%%Задание 4
#4.1.Считываем CSV с разделителем ';'
df = pd.DataFrame.from_csv("goalies-2014-2016.csv", sep = ';', index_col = None)
df.loc[:4,:'games_played'] #выводим первые 5 строки и первые 6 столбцов нашего набора данных

#%%
#4.2 Проверка Save_percentage
df['CheckSaves'] = round(df['saves'],3)/round(df['shots_against'],3) #совпадает
#проверка абсолютного наибольшего отклонения от табличного значения
maxi=abs(df['CheckSaves']-df['save_percentage']).max()
print(maxi)
#%%
#4.3 средние и стандартные отклония для games_played, goals_against, save_percetage
print("Means")
print("games_played: %s " % (df.games_played.mean()))
print("goals_against: %s " % (df.goals_against.mean()))
print("save_percentage: %s " % (df.save_percentage.mean()))
print('\n')
print("STDs")
print("games_played: %s " % (df.games_played.std()))
print("goals_against: %s " % (df.goals_against.std()))
print("save_percentage: %s " % (df.save_percentage.std()))
#%%
#4.4. вратарь с наибольшим количеством отражённых бросков, 40+матчей
date = df['season'] == '2016-17'
games = df['games_played'] > 40
saves = df['save_percentage']
a = df[date & games & saves]
b = a['save_percentage'].max()

temp = df.loc[(df['games_played']>40) & (df['season'] == '2016-17')& (df['save_percentage']==b)]
temp[['player', 'save_percentage']]

#%%
#4.5 наибольший saves по сезонам
kek = df.loc[((df['season'] == '2016-17'))]
lul1617 = kek['saves'].max()
abc = kek.loc[(kek['saves']==lul1617)]

kok = df.loc[((df['season'] == '2015-16'))]
lul1516 = kok['saves'].max()
de = kok.loc[(kok['saves']==lul1516)]

kuk = df.loc[((df['season'] == '2014-15'))]
lul1415 = kuk['saves'].max()
fhk = kuk.loc[(kuk['saves']==lul1415)]

print (pd.concat([abc, de, fhk])[["season", "player", "saves"]])

#%%
#4.6 
df1617 = df[(df['season'] == '2016-17') & (df['wins'] >= 30)]
df1516 = df[(df['season'] == '2015-16') & (df['wins'] >= 30)]
df1415 = df[(df['season'] == '2014-15') & (df['wins'] >= 30)]
frame = df1617.append([df1516, df1415])

newlist = {}
for i in range(len(frame)):
    if frame.iloc[i]['player'] in newlist:    
        Playername = frame.iloc[i]['player']
        newlist[Playername] += 1
    else:
        Playername = frame.iloc[i]['player']
        newlist[Playername] = 1

dfn = pd.DataFrame(columns = df.columns)

for i in newlist:
    if newlist[i] == 3:
        dfn = dfn.append(df[df['player'] == i])

print(dfn[['player', 'season', 'wins']])