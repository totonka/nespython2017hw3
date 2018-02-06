import numpy as np
from scipy.stats import t as T
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(0)
#Task 1
class OLS(object):
    
    def __init__(self, y, X):
        self.y = y
        self.X = X
        self.beta = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(y)
        n = y.shape[0]
        k = X.shape[1]
        self.sigmasqr = 1/(n - k) \
                        * (y - X.dot(self.beta)).T.dot(y - X.dot(self.beta))
        self.V = self.sigmasqr * np.linalg.inv(X.T.dot(X))
        
    def predict(self, x):
        prediction = x.T.dot(self.beta)
        X = self.X
        hpse = self.sigmasqr * (1 + x.T.dot(np.linalg.inv(X.T.dot(X))).dot(x)) 
        return (prediction, hpse)
    
def factorial(n):
    if n < 1:
        return 1
    else:
        f = 1
        for i in range(1, n + 1):
            f *= i
        return f
    
def Task2():
    #define constants
    N = 200
    K = 4
    k = 11
    alpha = 0.1
    a = -5
    b = 5
    sigmasqr = 100
    sigma = sigmasqr ** 0.5
    mu = 0
    precision = 6
    #since rand gives random numbers from [0; 1), to exclude 0 with some precision
    # we could add that small part to lower boundary, and make it (0; 1)
    epsilon = 10 ** (-precision)
    beta = epsilon + (1 - epsilon) * np.random.rand(k)
    #expand standard uniform interval
    a = a + epsilon
    x = a + (b - a) * np.random.rand(N)
    #sort x for meaningful plots
    x = np.sort(x)
    #switch from standard normal to normal
    u = sigma * np.random.randn(N) + mu
    y = np.empty(N)
    #array from 0 to 10 for powers and factorial
    k = np.arange(0, k, 1)
    #making custom factorial function to work element-wise for numpy array
    npFact = np.vectorize(factorial)
    for i in range(N):
        #array filled with x[i] used for operations with k, loop alternative
        y[i] = (np.full(11, x[i]) ** k / npFact(k)).sum() + u[i]
        
    models = {}
    #matrix for x 
    m = np.empty(shape = (N, 0))
    #[-5; 5]
    a = a - epsilon
    b = b + epsilon
    x2 = a + (b - a) * np.random.rand(N)
    x2 = np.sort(x2)
    # in cycle add additional x parameter to matrix and estimate OLS for each K
    for k in range(4):
        xk = x2 ** (k + 1)
        #add x as additional N x 1 column to matrix
        m = np.hstack((m, xk.reshape(N, 1)))
        models[k] = OLS(y - u, m)
        
    #array of predicted values
    preds = np.empty((K, N))
    #array of prediction errors
    errs = np.empty((K, N))
    for k in range(K):
        for i in range(200):
            #array of powers for x, + 2 - powers start from 1, k from zero
            #double shift
            power = np.arange(1, k + 2, 1)
            #array of x for predict function
            xi = np.full(k + 1, x2[i])
            preds[k][i] = models[k].predict(xi ** power)[0]
            errs[k][i] = models[k].predict(xi ** power)[1]
    t = T.ppf(1 - alpha / 2, N - 1) 
    higher = preds[3] + errs[3] * t
    lower = preds[3] - errs[3] * t

    #selection
    plt.figure(1)
    plt.scatter(x, y)
    plt.xlabel(r'$x_i$')
    plt.ylabel(r'$y_i$')

    #regressions for x [-5; 5]
    plt.figure(2) 
    plt.scatter(x2, y)
    plt.plot(x2, preds[0], color = 'blue', label = 'K = 1')
    plt.plot(x2, preds[1], color = 'orange', label = 'K = 2')
    plt.plot(x2, preds[2], color = 'green', label = 'K = 3')
    plt.plot(x2, preds[3], color = 'red', label = 'K = 4')
    plt.xlabel(r'$x_i$')
    plt.ylabel(r'$y_i$')
    plt.legend(loc = 2)

    #conf interval
    plt.figure(3)
    plt.scatter(x2, y)
    plt.plot(x2, preds[3], color = 'red', label = 'K = 4')
    plt.fill_between(x2, higher, lower, facecolor = 'grey', alpha = 0.3)
    plt.xlabel(r'$x_i$')
    plt.ylabel(r'$y_i$')
    plt.legend(loc = 2)
    
    plt.show()
def Task3():
    conf = 0.1
    N = 100
    M = np.random.randn(N, N)
    t = T.ppf(1 - conf / 2, N - 1)
    #Columns
    colMean = M.mean(axis = 0)
    colStd = M.std(axis = 0)
    colHigh = colMean + t * colStd
    colLow = colMean - t * colStd
    colCondition = np.logical_and(colHigh >= 0, colLow <= 0)
    colTrue = colCondition[colCondition == True].size
    #1
    print(colCondition)
    #2
    print(colTrue)
    #Rows
    rowMean = M.mean(axis = 1)
    rowStd = M.std(axis = 1)
    rowHigh = rowMean + t * rowStd
    rowLow = rowMean - t * rowStd
    rowCondition = np.logical_and(rowHigh >= 0, rowLow <= 0)
    rowTrue = rowCondition[rowCondition == True].size
    #1
    print(rowCondition)
    #2
    print(rowTrue)
    pass
    
def Task4():
    df = pd.read_csv('goalies-2014-2016.csv', sep = ';')
    #1
    print(df.iloc[:5, :6])
    sp = df['saves']/df['shots_against']
    precision = (np.abs(df['save_percentage'] - sp)).max()
    #2
    print(precision)
    #3
    print(df[['games_played', 'goals_against', 'save_percentage']].mean())
    print(df[['games_played', 'goals_against', 'save_percentage']].std())
    #4
    print(df[(df['season'] == '2016-17') & (df['games_played'] > 40)].\
          loc[:, ['player', 'save_percentage']].sort_values('save_percentage').\
          tail(1))
    #5
    print((df.assign(rn = df.sort_values(['saves'], ascending=False)
                         .groupby(['season'])
                         .cumcount() + 1)
            .query('rn == 1')
            .sort_values(['season'], ascending = False)
        ).loc[:, ['player', 'season', 'saves']])
    #6
    wins30 = df[df['wins'] >= 30].loc[:,['player','season','wins']].\
             groupby('player').agg({'season' : np.size, 'wins' : np.sum})
    print(wins30[wins30['season'] > 2])
    pass

if __name__ == '__main__':
    Task3()
    Task4()
    Task2()
