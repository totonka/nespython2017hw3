import numpy as np
from numpy import matmul as mm

#%% 
################# 1

X = np.random.randn(100,3)
y = X.dot(np.array([1, 2, 3])) + np.random.randn(100)

class OLS:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.beta = mm(mm(np.linalg.inv(mm(X.T, X)),X.T), y)
        self.sigma = (1/(X.shape[0] - X.shape[1])) * mm((y - mm(X, self.beta)).T, (y - mm(X, self.beta)))
        self.V = self.sigma * np.linalg.inv(mm(X.T, X))

    def predict(self, x):
        x = np.array([1, 0, 1])
        prediction = np.array([mm(x.T, self.beta), self.sigma * (1 + mm(mm(x.T, np.linalg.inv(mm(X.T, X))), x))])
        return(prediction[0], prediction[1])		

#%%
model = OLS(X, y)
#%%
model.beta
#%%
model.V
#%%
model.predict(np.array([1,0,1]))
#%%
######################2

import matplotlib 
from matplotlib import pyplot as plt
import scipy.stats as st

b = np.random.rand(11)
x = np.random.randn(200,1)
for i in range(len(x)):
    x[i, 0] = np.random.uniform(-3, 7)
	
y = u = 100 * np.random.randn(200)

for i in range(200):
    for k in range(11):
        y[i] += b[k] * (x[i,0] ** k) / np.math.factorial(k)
#%%		
plt.figure(num=None, figsize=(9, 7), facecolor='w', edgecolor='k')
plt.scatter(x, y, s = 3)

#%%

x0 = x ** 0
x2 = x ** 2
x3 = x ** 3
x4 = x ** 4
X1 = np.concatenate((x0, x),axis=1)
X2 = np.concatenate((X1, x2),axis=1)
X3 = np.concatenate((X2, x3),axis=1)
X4 = np.concatenate((X3, x4),axis=1)
model_X1 = OLS(X1, y)
model_X2 = OLS(X2, y)
model_X3 = OLS(X3, y)
model_X4 = OLS(X4, y)
model_X4.beta

t = np.linspace(-3, 7, 100)
line1 = np.zeros(len(t))
for i in range(len(t)):
    line1[i] = model_X1.beta[0] + model_X1.beta[1] * t[i]
	
t = np.linspace(-3, 7, 100)
line2 = np.zeros(len(t))
for i in range(len(t)):
    line2[i] = model_X2.beta[0] + model_X2.beta[1] * t[i] + model_X2.beta[2] * t[i]**2
	
t = np.linspace(-3, 7, 100)
line3 = np.zeros(len(t))
for i in range(len(t)):
    line3[i] = model_X3.beta[0] + model_X3.beta[1] * t[i] + model_X3.beta[2] * t[i]**2 + model_X3.beta[3] * t[i]**3
	
#%%	
plt.figure(num=None, figsize=(9, 7), facecolor='w', edgecolor='k')
plt.scatter(x, y, s = 3)
plt.plot(t, line1, label = 'K = 1')
plt.plot(t, line2, label = 'K = 2')
plt.plot(t, line3, label = 'K = 3')
plt.plot(t, line4, label = 'K = 4')

#%%

plt.figure(num=None, figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(x, y, s = 3)
plt.plot(t, line4)
plt.fill_between(t, line4 - 120, line4 + 120, alpha = 0.5)

#%%

####################3

import scipy.stats as st
#%%
A = np.random.randn(100, 100)

#%%

# columns
count = 0
flags = []
for column in A.T:
    a = st.t.interval(0.90, len(column) - 1, loc = np.mean(column), scale = st.sem(column))
    if (a[0] < 0 < a[1]):
        flags.append(True)
        count += 1
    else:
        flags.append(False)
        
print(flags)
print(count)

#%%


# rows
count = 0
flags = []
for column in A:
    a = st.t.interval(0.90, len(column) - 1, loc = np.mean(column), scale = st.sem(column))
    if (a[0] < 0 < a[1]):
        flags.append(True)
        count += 1
    else:
        flags.append(False)
        
print(flags)
print(count)

#%%

#########################4

import pandas as pd

#%%

df = pd.read_csv('goalies-2014-2016.csv', delimiter = ';')

#%%

df.iloc[0:5, 0:6]

#%%

max = (abs((df['saves']) / df['shots_against'] - df['save_percentage'])).max()
max

#%%
for mark in ['games_played', 'goals_against', 'save_percentage']:
    print( mark + '\t', df[mark].mean())
	
#%%	
for mark in ['games_played', 'goals_against', 'save_percentage']:
    print( mark + '\t', df[mark].std())

#%%	
best = pd.DataFrame(df[['player', 'save_percentage']].iloc[df[(df.season=='2016-17') & (df.games_played > 40)].save_percentage.idxmax()])
best_16_17_40 = best.T

#%%

best_16_17_40

#%%


df.sort_values('saves', ascending = False).drop_duplicates(['season'])[['season', 'player', 'saves']].sort_values('season')

#%%

goalkeepers = df[df.wins >= 30][['season', 'player', 'wins']].groupby(['season', 'player']).sum()
best_goalkeepers = goalkeepers.groupby('player').sum()
best_goalkeepers['number_of_seasons'] = goalkeepers.groupby('player').count()


best_goalkeepers[best_goalkeepers.number_of_seasons >= 3]

