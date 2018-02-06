#HW3
#я решил не писать отчет, так как вроде все и так понятно

#%%
#Problem 1

import numpy as np
n=3
k=2
x = np.random.randn(n,k)
y = x.dot(np.random.randn(k))+np.random.randn(n)

class OLS(object):
    def __init__(self,a,b):
        self.a = a
        self.b = b
        #self.beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(b),b)),np.transpose(b)),a)
        self.beta = (np.linalg.inv(b.transpose() @ b)) @ b.transpose() @ a
       # self.V = ((np.transpose(a-b.dot(self.beta)).dot(a-b.dot(self.beta))).dot(np.linalg.inv(np.transpose(b).dot(b))))/(len(a)-b.shape[1])
       # self.V = ((((a-b @ self.beta).transpose()) * (a-b @ self.beta)) @ (np.linalg.inv((b.transpose())@ b)))/(a.shape[0]-b.shape[1])
        self.V = (a - b @ self.beta).transpose() @ (a - b @ self.beta) * np.linalg.inv(b.transpose() @ b)/(a.shape[0]-b.shape[1])
    def predict(self, t):
        self.t = t
        print("Предсказание: %s, Ошибка: %s " % ((self.t).transpose() @ self.beta, (self.a - self.b @ self.beta).transpose() @ (self.a - self.b @ self.beta) *(1+(self.t).transpose() @ np.linalg.inv((self.b).transpose() @ self.b) @ self.t )))

model = OLS(y, x)
model.beta
model.V
model.predict(np.random.randn(k))


#%%
#Problem 2
#(2.1)
beta = np.random.rand(11)
beta

#(2.2)

x = (np.random.rand(200) * 10 - 5)

#(2.3)
import math
u = (np.random.randn(200) * math.sqrt(100))

#(2.4)
p = np.zeros(200)
y = np.zeros(200)
for k in range (0,10):
        p = p + ((beta[k]) * (x ** k))/math.factorial(k)
        k = k + 1
y = p + u
print(y)

#(2.5)
import matplotlib.pyplot as plt
plt.scatter(x,y)
#plt.title("title")
plt.xlabel("xi")
plt.ylabel("yi")
plt.show()






x_0 = np.ones(200)
x_1 = x 
x_2 = x ** 2
x_3 = x ** 3
x_4 = x ** 4
    
r_1 = np.column_stack((x_0,x_1))
r_2 = np.column_stack((r_1,x_2))
r_3 = np.column_stack((r_2,x_3))
r_4 = np.column_stack((r_3,x_4))

n=200
k=3
#x = np.random.randn(n,k)
#y = x.dot(np.random.randn(k))+np.random.randn(n)

class OLS(object):
    def __init__(self,a,b):
        self.a = a
        self.b = b
        #self.beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(b),b)),np.transpose(b)),a)
        self.beta = (np.linalg.inv(b.transpose() @ b)) @ b.transpose() @ a
       # self.V = ((np.transpose(a-b.dot(self.beta)).dot(a-b.dot(self.beta))).dot(np.linalg.inv(np.transpose(b).dot(b))))/(len(a)-b.shape[1])
       # self.V = ((((a-b @ self.beta).transpose()) * (a-b @ self.beta)) @ (np.linalg.inv((b.transpose())@ b)))/(a.shape[0]-b.shape[1])
        self.V = (a - b @ self.beta).transpose() @ (a - b @ self.beta) * np.linalg.inv(b.transpose() @ b)/(a.shape[0]-b.shape[1])
    def predict(self, t):
        self.t = t
        #print("Предсказание: %s, Ошибка: %s " % ((self.t).transpose() @ self.beta, (self.a - self.b @ self.beta).transpose() @ (self.a - self.b @ self.beta) *(1+(self.t).transpose() @ np.linalg.inv((self.b).transpose() @ self.b) @ self.t )))
        print(self.t @ self.beta)
        plt.scatter(x,self.t @ self.beta)

        
        
        
        
model1 = OLS(y, r_1)
model2 = OLS(y, r_2)
model3 = OLS(y, r_3)
model4 = OLS(y, r_4)
t1 = r_1 @ model1.beta
t2 = r_2 @ model2.beta
t3 = r_3 @ model3.beta
t4 = r_4 @ model4.beta



plt.scatter(x,y)
#plt.title("title")
plt.scatter(x,t1,color = 'blue')
plt.scatter(x,t2,color = 'orange')
plt.scatter(x,t3,color = 'green')
plt.scatter(x,t4,color = 'red')
#plt.plot(t3)
plt.xlabel("xi")
plt.ylabel("yi")
plt.show()


#%%
plt.plot(np.sort(x,t2,axis=1),color = 'red')

#%%
#Problem 3.1 (Для столбцов)

import numpy as np
import matplotlib
import scipy.stats
import math

mean = np.zeros(100)
cov = np.eye(100)
x = np.random.multivariate_normal(mean, cov, 100)
a = [[0,0]] * 100

for i in range (0,99):
    a[i] = [np.mean(x[:,i]) - scipy.stats.t.ppf(0.9,99) * np.std(x[:,i])/math.sqrt(100), np.mean(x[:,i]) + scipy.stats.t.ppf(0.9,99) * np.std(x[:,i])/math.sqrt(100)]
     
    i = i + 1

for j in range (0,99):
    print("L %s R %s " % (a[j][0],a[j][1]))
    j = j + 1

matplotlib.pyplot.plot(a)
p=[]
t=0
for k in range (0,99):
    if a[k][0]<0 and a[k][1]>0:
        p.append(True)
        t=t+1
    else: 
        p.append(False)
print((p,t))


#%%

#Problem 3.2 (Для строк)

import numpy as np
import matplotlib
import scipy.stats
import math

mean = np.zeros(100)
cov = np.eye(100)
x = np.random.multivariate_normal(mean, cov, 100)
a = [[0,0]] * 100

for i in range (0,99):
    a[i] = [np.mean(x[i,:]) - scipy.stats.t.ppf(0.9,99) * np.std(x[i,:])/math.sqrt(100), np.mean(x[i,:]) + scipy.stats.t.ppf(0.9,99) * np.std(x[i,:])/math.sqrt(100)]
     
    i = i + 1

for j in range (0,99):
    print("L %s R %s " % (a[j][0],a[j][1]))
    j = j + 1

matplotlib.pyplot.plot(a)
p=[]
t=0
for k in range (0,99):
    if a[k][0]<0 and a[k][1]>0:
        p.append(True)
        t=t+1
    else: 
        p.append(False)
print((p,t))

#%%

#Problem 4

import pandas as pd
#import matplotlib.pyplot as plt

(4.1)
data = pd.read_csv('goalies-2014-2016.csv', sep=';') 
data1 = data.iloc[0:5,0:6]
data1

#%%
#(4.2) 

data_saves = data['saves']
data_shots_against = data['shots_against']
data_save_percentage = data['save_percentage']
data_save_percentage_calc = data_saves/data_shots_against
data_abs_dev = data_save_percentage_calc-data_save_percentage
data_m = max(data_abs_dev)
data_m

#%%
#(4.3)

data_games_played = data['games_played']
data_goals_against = data['goals_against']
print("Средние: %s  %s  %s" % (data_games_played.mean(),
data_goals_against.mean(),
data_save_percentage.mean()))

print("Отклонения: %s  %s  %s" % (data_games_played.std(),
data_goals_against.std(),
data_save_percentage.std()))

#%%
#(4.4)


data_16_17 = data[data['season'] == '2016-17']
data_games_played = data_16_17[data_16_17['games_played'] > 40]
data_games_played_sorted = data_games_played.sort_values(by=['save_percentage'],ascending = False)
data_fin = data_games_played_sorted[['player','save_percentage']]
#data_finn = data_fin.iloc[0:2]
d = data_fin[:1]
d

#%%
#(4.5)

data_16_17 = data[data['season'] == '2016-17']
data_15_16 = data[data['season'] == '2015-16']
data_14_15 = data[data['season'] == '2014-15']
d_og17 = data_16_17[['season','player','saves']]
d_og16 = data_15_16[['season','player','saves']]
d_og15 = data_14_15[['season','player','saves']]
d_og17_sorted = d_og17.sort_values(by=['saves'],ascending = False)
d_og16_sorted = d_og16.sort_values(by=['saves'],ascending = False)
d_og15_sorted = d_og15.sort_values(by=['saves'],ascending = False)
d17 = d_og17_sorted[:1]
d16 = d_og16_sorted[:1]
d15 = d_og15_sorted[:1]
print("%s\n  %s\n  %s" % (d17,
d16,
d15))


#%%
#(4.6)

data_wins = data [data['wins'] >= 30]
data_wins_ogr = data_wins[['season','player','wins']]
data_wins_ogr_sorted = data_wins_ogr.sort_values(by=['player'],ascending = False)
c = pd.value_counts(data_wins_ogr_sorted['player'].values, sort=True)
c[:6]


#for i in range (0,len(data_wins_ogr)):
 #   if data_wins_ogr_sorted[[]]
    







