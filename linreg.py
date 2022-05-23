import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionUsingGD:

    def __init__(self, eta=0.05, n_iterations=1000):
        self.eta = eta
        self.n_iterations = n_iterations

    def fit(self, x, y):
        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.eta / m) * gradient_vector
            cost = np.sum((residuals ** 2) / (2 * m))
            self.cost_.append(cost)
        return self

    def predict(self, x):
        return np.dot(x, self.w_)


np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)
x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
lin_reg = LinearRegressionUsingGD()
lin_reg.fit(x, y)
plt.plot(x, lin_reg.predict(x))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
