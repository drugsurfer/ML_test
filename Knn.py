from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

data = pd.DataFrame({'x1': [-1, 1, 2, -3, 3], 'x2': [0, 2, -2, -1, 2], 'y': [1, 0, 0, 1, 1]},
                    index=['A', 'B', 'C', 'D', 'E'])

y_test = pd.DataFrame({'x1': [0], 'x2': [0]})

X = data.drop(['y'], axis=1)
y = data.y

clf = KNeighborsClassifier(n_neighbors=5)  # тут меняет количество соседей
clf.fit(X, y)
print(clf.predict(y_test))


