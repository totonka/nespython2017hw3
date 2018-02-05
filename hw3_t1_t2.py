import numpy as np
import scipy.stats
from matplotlib import pyplot

class OLS:

    def __init__(self, y, X):
        self.dep_var = y
        self.indep_var = X

        self.beta  = np.linalg.pinv(X).dot(y)
        n, k = X.shape[0], X.shape[1]
        self.sigma = 1.0/(n-k)*np.linalg.norm(y - np.dot(X, self.beta),ord=2)**2
        self.VCV_beta = self.sigma*np.linalg.inv(np.dot(X.T, X))

    def predict(self, nX):

        return np.dot(nX, self.beta)

    def prediction_interval(self, nX):

        ystd = np.diagonal(self.sigma*(1.0+nX.dot(np.linalg.inv(np.dot(self.indep_var.T, self.indep_var))).dot(nX.T)))
        ci_u = self.predict(nX) + ystd**0.5 * scipy.stats.t.ppf(0.95, self.indep_var.shape[0]-self.indep_var.shape[1])
        ci_l = self.predict(nX) - ystd**0.5 * scipy.stats.t.ppf(0.95, self.indep_var.shape[0]-self.indep_var.shape[1])

        return ci_u, ci_l

def _main():

    beta = np.random.rand(11)
    x = 10*np.random.rand(200)-5.0
    u = 10 * np.random.randn(200)
    y = np.zeros(200)
    for k in range(11):
        y = y + beta[k]*x**k / np.math.factorial(k)
    y = y + u

    pyplot.figure(1)
    pyplot.scatter(x,y)
    pyplot.xlabel('$x_i$')
    pyplot.ylabel('$y_i$')

    pyplot.savefig('1.png', bbox_inches='tight')

    X      = np.empty([200, 5])
    X_grid = np.empty([200, 5])
    Y_pred = np.empty([200, 4])

    models = list()

    pyplot.figure(2)
    for k in range(5):
        X[:,k] =  x ** k
        X_grid[:,k] = np.arange(-5, 5, 10.0/200) ** k
    for k in range(1,5):
        models.append(OLS(y,X[:,0:k+1]))
        Y_pred[:,k-1] = models[k-1].predict(X_grid[:,0:k+1])
        pyplot.plot(np.arange(-5, 5, 10.0/200), Y_pred[:,k-1], label = 'k = {}'.format(str(k)))

    pyplot.scatter(x, y, s = 4)
    pyplot.xlabel('$x_i$')
    pyplot.ylabel('$y_i$')
    pyplot.legend()
    pyplot.savefig('2.png', bbox_inches='tight')
    #pyplot.show()

    pyplot.figure(3)

    pi_u, pi_l = models[3].prediction_interval(X_grid[:,0:5])
    pyplot.plot(np.arange(-5, 5, 10.0 / 200), Y_pred[:, 3], label='k = {}'.format(str(4)))
    pyplot.fill_between(np.arange(-5, 5, 10.0/200), pi_l, pi_u,alpha = 0.3)
    pyplot.scatter(x, y, s=4)
    pyplot.xlabel('$x_i$')
    pyplot.ylabel('$y_i$')
    pyplot.legend()
    #pyplot.plot(np.arange(-5, 5, 10.0 / 200), pi_l)
    pyplot.savefig('3.png', bbox_inches='tight')
    pyplot.show()

if __name__ == "__main__":
    _main()