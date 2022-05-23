import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import datasets


class Kmeans:
    def __init__(self):
        pass

    def kmeans_fit_predict(self, x, k=10, max_iter=1000, tol=0.1, low=0.0, high=1.0):
        centroids = np.random.uniform(low, high, size=(k, x.shape[1]))
        cent_history, sum_dist = [], []
        cent_history.append(centroids)
        for i in range(max_iter):
            labels = np.argmin(cdist(x, cent_history[i], metric='euclidean'), axis=1)
            new_centroids = centroids.copy()
            sum_ = 0
            for j in range(new_centroids.shape[0]):
                if j not in labels:
                    for k in range(new_centroids.shape[1]):
                        new_centroids[j, k] = np.random.uniform(low, high, size=1)[0]
                    sum_ += 0
                else:
                    for k in range(new_centroids.shape[1]):
                        new_centroids[j, k] = np.mean(x[labels==j, k])
                    sum_ += np.sum(cdist(x[labels==j], cent_history[i][j].reshape(1, new_centroids.shape[1])))
            cent_history.append(new_centroids)
            sum_dist.append(sum_)
            if i != 0:
                print(abs(sum_dist[i] - sum_dist[i - 1]))
            if i != 0 and abs(sum_dist[i] - sum_dist[i - 1]) < tol:
                break
        return cent_history[-1], labels


X, y = datasets.load_digits(return_X_y=True)

kmeans = Kmeans()
clusters, labels_mnist = kmeans.kmeans_fit_predict(X, k=len(np.unique(y)),low=0.0, high=np.max(X))
number = 3
count = 10
testX = X[labels_mnist == number]
testX[0, :].reshape([8, 8])
f, axes = plt.subplots(1, count, sharey=True, figsize=(40, 6))
for i in range(count):
    axes[i].imshow(testX[i, :].reshape([8, 8]), cmap='gray')
plt.show()


