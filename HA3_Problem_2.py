import numpy as np
from math import factorial, sqrt
import scipy.stats as ss
import matplotlib.pyplot as plt


class OLS():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        Xt = X.T
        XX = (Xt @ X)
        XX1 = np.linalg.inv(XX)
        self.XX1 = XX1
        XX1X = XX1 @ Xt
        beta = XX1X @ Y
        self.beta = beta
        n = X.shape[0]
        k = X.shape[1]
        yXb = Y - X @ beta
        yXbt = yXb.T
        sigma2 = 1 / (n - k) * yXbt @ yXb
        self.sigma2 = sigma2
        varBeta = sigma2 * XX1
        self.V = varBeta

    def predict(self, X):
        yPred = X.T @ self.beta
        varPred = self.sigma2 * (1 + X.T @ self.XX1 @ X)
        return (yPred, varPred)


#Task_1
beta = np.random.rand(11)

#Task_2
x = np.random.uniform(-5.0, 5.0, 200)

#Task_3
U = 10 * np.random.randn(200,)  # sigma * np.random.randn(...) + mu

#Task_4
y = []
for i in range(0, 200):
    sm = 0
    for k in range(11):
        sm += (beta[k] * (x[i]) ** k / factorial(k))
    sm += U[i]
    y.append(sm)

y = np.array(y)

#Task_5
plt.scatter(x, y)
plt.xlabel('$x_i$')
plt.ylabel('$y_i$')
#plt.savefig('fig_11.svg')
plt.show()

#OLS
t_stat = ss.t.ppf(0.95, 195)
#print(t_stat)

x.shape = (200, 1)
unit_x = np.ones((200, 1), dtype=int)
xx = np.concatenate((unit_x, x), axis=1)

res = OLS(xx, y)

a = np.arange(-5, 5, 0.1)
a.shape = (100, 1)
unit_a = np.ones((100, 1), dtype=int)
aa = np.concatenate((unit_a, a), axis=1)

grafX = []
grafY = []
dispYb = []
dispYt = []
for j in aa:
    grafY.append(res.predict(j)[0])
    grafX.append(j[1])

graf = [(grafX, grafY)]
for i in range(2, 5):
    xk = np.copy(x)**i
    xx = np.concatenate((xx, xk), axis=1)
    res = OLS(xx, y)

    ak = np.copy(a)**i
    aa = np.concatenate((aa, ak), axis=1)
    grafX = []
    grafY = []
    dispYb = []
    dispYt = []
    for j in aa:
        grafY.append(res.predict(j)[0])
        grafX.append(j[1])
        if i == 4:
            dispYb.append(res.predict(j)[0] - t_stat * sqrt(res.predict(j)[1]))
            dispYt.append(res.predict(j)[0] + t_stat * sqrt(res.predict(j)[1]))
    graf.append((grafX, grafY))

disp = [(dispYb, dispYt)]

plt.scatter(x, y)
plt.plot(graf[0][0], graf[0][1], 'b-', label='K = 1')
plt.plot(graf[1][0], graf[1][1], 'y-', label='K = 2')
plt.plot(graf[2][0], graf[2][1], 'g-', label='K = 3')
plt.plot(graf[3][0], graf[3][1], 'r-', label='K = 4')
plt.legend(loc='best')
plt.xlabel('$x_i$')
plt.ylabel('$y_i$')
#plt.savefig('fig_22.svg')
plt.show()


#Confident_interval
plt.scatter(x, y)
plt.plot(graf[3][0], graf[3][1], 'r-', label='K = 4')
plt.fill_between(graf[3][0], disp[0][0], disp[0][1], facecolor='gray', alpha='0.5')
plt.legend(loc='upper left')
plt.xlabel('$x_i$')
plt.ylabel('$y_i$')
#plt.savefig('fig_33.png')
#plt.savefig('fig_33.svg')
#plt.savefig('fig_3.pdf')
plt.show()
