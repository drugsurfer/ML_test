import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist


def euclidian_metric(a, b):
    return np.sqrt(np.sum(np.power(a - b, 2), axis=1))


def find_neighbours(k, y, distances):
    # функция, которая по уже рассчитанным расстояниям от точек y до нашей точки (distances)
    # находит и возвращает k ближайших соседей и соответствующие расстояния до них
    distances = np.array(distances).reshape(1, distances.shape[0])
    sort_distances = np.argsort(distances)
    neighbours = []
    neighbours_distances = []
    for i in range(k):
        neighbours.append(y[sort_distances[0][i]])
        neighbours_distances.append(distances[0][sort_distances[0][i]])
    return neighbours, neighbours_distances


def get_closest_classes(neighbours):
    # функция, выделяющую преобладающий класс среди соседних объектов.
    # Если есть конкурирующие классы, то нужно вернуть их все
    unique_elements, count_elements = np.unique(neighbours, return_counts=True)
    max_value = unique_elements[np.where(count_elements == np.max(count_elements))]
    return max_value


def choose_best_class(best_classes, neighbours, neighbouring_distances):
    # функция, которая выбирает наиболее подходящий класс, при условии,
    # что есть несколько конкурирующих классов
    neighbours = np.array(neighbours)
    neighbouring_distances = np.array(neighbouring_distances)
    min_mean_dist = np.inf
    best_class = None
    for c in best_classes:
        temp = np.min(neighbouring_distances[np.where(neighbours == c)])
        if temp < min_mean_dist:
            min_mean_dist = temp
            best_class = c
    return best_class


def nearest_neighbours_classify(x, y, k, x_pred):
    # эта функция будет считать расстояния каждой точки элемента выборки x_pred до всех точек
    # в исходном датасете X и на основе расстояний вычислять принадлежность к классу
    res = np.zeros(x_pred.shape[0], dtype=y.dtype)
    for i in range(x_pred.shape[0]):
        distances = cdist(x, x_pred[i].reshape(1, x_pred.shape[1]), metric='euclidean')  # считаем расстояния до классов
        neighbours, neighbouring_distances = find_neighbours(k, y, distances)  # находим ровно k соседей этой точки
        best_classes = get_closest_classes(neighbours)  # обнаруживаем классы, которые имеются среди соседей
        res[i] = choose_best_class(best_classes, neighbours,
                                   neighbouring_distances)  # выбираем наиболее релевантный класс по среднему расстоянию до него среди соседей
    return res


np.random.seed(seed=42)
'''
p1 = np.random.normal(loc=0, scale=1, size=(50,2))
p2 = np.random.normal(loc=5, scale=2, size=(50,2))
p3 = np.random.normal(loc=10, scale=0.8, size=(50,2)) - np.array([5, -5])

X = np.concatenate((p1, p2, p3))
y = np.array([1]*50 + [2]*50 + [3]*50)

point = [2, 2.5]

plt.scatter(p1[:,0], p1[:, 1], color='blue')
plt.scatter(p2[:,0], p2[:, 1], color='orange')
plt.scatter(p3[:,0], p3[:, 1], color='green')
plt.scatter(point[0], point[1], s = 100, color='red')
plt.show()
'''
'''
Y = np.arange(10)
dist = np.linspace(1, 10, 10)
neighbours = find_neighbours(3, Y, dist)
real_neighbours = ([0, 1, 2], [1., 2., 3.])
for i in range(len(neighbours)):
    for j in range(len(neighbours[i])):
        assert neighbours[i][j] == real_neighbours[i][j]
print('Если вы видите этот текст, но не видите ошибку, то всё работает корректно!')
'''
'''
assert get_closest_classes(np.asarray([1,2,3,2,2])) == [2]
closest = get_closest_classes(np.asarray([1,2,3,2,3]))
assert closest[0] == 2 and closest[1] == 3
print('get_closest_classes работает верно')
'''
'''
assert choose_best_class([1,2], np.array([1, 2, 1, 3, 2]), np.asarray([0.5, 1, 1, 8, 0.6])) == 1
print('Проверка пройдена')
'''
p1 = np.random.normal(loc=0, scale=1, size=(50,2))
p2 = np.random.normal(loc=5, scale=2, size=(50,2))
p3 = np.random.normal(loc=10, scale=0.8, size=(50,2)) - np.array([5, -5])

X = np.concatenate((p1, p2, p3))
y = np.array([1]*50 + [2]*50 + [3]*50)
# train_test_split разбивает X и y на выборки при этом размер тестовой выборки устанавливается через test_size
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# выберем число соседей =  5
# для данных из тестовой выборки предскажем, к какому классу относятся точки
y_pred = nearest_neighbours_classify(X_train, y_train, 5, X_test)

print(accuracy_score(y_test, y_pred))

