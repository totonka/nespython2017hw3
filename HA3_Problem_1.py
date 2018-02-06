import numpy as np


class OLS():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        Xt = X.T
        XX = (Xt @ X)
        XX1 = np.linalg.inv(XX)
        self.XX1 = XX1
        XX1X = XX1 @ Xt
        beta = XX1X @ y
        self.beta = beta
        n = X.shape[0]
        k = X.shape[1]
        yXb = y - X @ beta
        yXbt = yXb.T
        sigma2 = 1 / (n - k) * yXbt @ yXb
        self.sigma2 = sigma2
        varBeta = sigma2 * XX1
        self.V = varBeta

    def predict(self, X):
        yPred = X.T @ self.beta
        varPred = self.sigma2 * (1 + X.T @ self.XX1 @ X)
        return (yPred, varPred)


x = np.random.randn(100, 3)
y = x.dot(np.array([1, 2, 3]))+np.random.randn(100)
c = OLS(x, y)
a = np.array([1, 0, 1])
print(c.beta, c.V, c.predict(a), sep='\n\n')
