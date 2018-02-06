#%%
#1
import numpy as np
X = np.random.randn(100,3)
Y = X.dot(np.array([1,2,3]))+np.random.randn(100)
            
class OLS(object):
    def __init__(self,y,x):
        self.y = y
        self.x = x
        xt = np.transpose(self.x)
        inv = np.linalg.inv(np.dot(xt,self.x))
        self.beta = np.dot(np.dot(inv,xt),self.y)
        var = (1/(self.x.shape[0]-self.x.shape[1]))*np.dot(np.transpose(self.y - np.dot(self.x,self.beta)),(self.y - np.dot(self.x,self.beta)))
        self.V = var*inv
        
    def predict(self,value):
        var = (1/(self.x.shape[0]-self.x.shape[1]))*np.dot(np.transpose(self.y - np.dot(self.x,self.beta)),(self.y - np.dot(self.x,self.beta)))
        xt = np.transpose(self.x)
        xt1 = np.transpose(value)
        inv = np.linalg.inv(np.dot(xt,self.x))
        pred = np.dot(xt1,self.beta)
        err = var*(1+np.dot(np.dot(xt1,inv),value))
        return (pred,err)

model= OLS(Y,X)
print(model.beta,'\n',model.V,'\n',model.predict(np.array([1,0,1])))
#%%
 #2
import math as mt
import matplotlib 
k = np.random.rand(11,1)
x = np.random.uniform(-5,5,(200,1))   
ones = np.ones([200,1])   
u =10*np.random.randn(200) 
x2 = np.power(x,2)
x3 = np.power(x,3)
x4 = np.power(x,4)
y = np.zeros((200,1))
for i in range(len(y)):
    y[i] += u[i]
    for j in range (len(k)):
        y[i] +=k[j]*np.power(x[i],j)/mt.factorial(j)  
x = np.concatenate((ones,x),1) 
mod1 = OLS(y,x)
x21 = np.concatenate((x,x2),1)
mod2 = OLS(y,x21)
x31 = np.concatenate((x,x2,x3),1)
mod3 = OLS(y,x31)
x41 = np.concatenate((x,x2,x3,x4),1)
mod4 = OLS(y,x41) 
res1 =np.zeros((200,))
res2 =np.zeros((200,))
res3 =np.zeros((200,))
res4 =np.zeros((200,))
for i in range(len(res1)):
    res1[i] = mod1.predict(x[i])[0]
for i in range(len(res2)):
    for j in range(len(mod2.beta)):   
        res2[i] += mod2.beta[j]*x21[i][j]
for i in range(len(res3)):
    res3[i] = mod3.predict(x31[i])[0]
for i in range(len(res4)):
    res4[i] = mod4.predict(x41[i])[0]
x = x[:,1]
new_x1, new_y1 = zip(*sorted(zip(x,res1)))
new_x2, new_y2 = zip(*sorted(zip(x,res2)))
new_x3, new_y3 = zip(*sorted(zip(x,res3)))
new_x4, new_y4 = zip(*sorted(zip(x,res4)))
matplotlib.pyplot.scatter(x,y)    
matplotlib.pyplot.plot(new_x1,new_y1,label = 'K=1')
matplotlib.pyplot.plot(new_x2,new_y2,label = 'K=2')
matplotlib.pyplot.plot(new_x3,new_y3,label = 'K=3')
matplotlib.pyplot.plot(new_x4,new_y4,label = 'K=4')
plt.pyplot.legend()

#%%
#3
import scipy.stats
a = np.random.randn(100,100) 
stat = []
number = 0
for i in range(100):   
    confh = np.mean(a[i])+np.std(a[i])/99**(1/2)*scipy.stats.t.ppf(0.95,99)
    confl = np.mean(a[i])-np.std(a[i])/99**(1/2)*scipy.stats.t.ppf(0.95,99)
    stat.append( confl<0<confh)
    if stat[i]==True:
        number+=1
print(stat,'\n',number)
stat = []
number = 0
for i in range(100):   
    confh = np.mean(a[:,i])+np.std(a[:,i])/99**(1/2)*scipy.stats.t.ppf(0.95,99)
    confl = np.mean(a[:,i])-np.std(a[:,i])/99**(1/2)*scipy.stats.t.ppf(0.95,99)
    stat.append( confl<0<confh)
    if stat[i]==True:
        number+=1
print(stat,'\n',number)

#%%
#4
import pandas as pd
x = pd.read_csv('goalies-2014-2016.csv',sep=';')
x.ix[:4,:6]

max(abs(x['save_percentage']-x['saves']/x['shots_against']))

x[['games_played','goals_against','save_percentage']].mean()
x[['games_played','goals_against','save_percentage']].std()

x[['player','save_percentage']][x.season=='2016-17'][x.games_played>40].nlargest(1,'save_percentage')

x.groupby('season')[['player','saves']].apply(lambda grp: grp.nlargest(1,'saves'))

