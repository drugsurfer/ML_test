import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


def standard_scaler(X):
    mean = X - np.mean(X, axis=0)
    return mean / np.std(X)


np.random.seed(0)
data = pd.read_csv('train.csv')
plt.figure(figsize=(12, 5))
x = data.drop(['count', 'datetime'], axis=1)
y = data['count']
x_scaled = standard_scaler(x)
x = x_scaled.to_numpy()
y = np.array([[i] for i in y.values])

x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
lin_reg = LinearRegressionUsingGD()
lin_reg.fit(x, y)
pred = lin_reg.predict(x)

RSS = ((y - pred) ** 2).sum()
TSS = ((y - y.mean()) ** 2).sum()

score = 1 - RSS / TSS