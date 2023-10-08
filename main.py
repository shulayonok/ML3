from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
import random


def design_matrix(function, X):
    F = np.zeros((len(X), 10))
    for i in range(len(X)):
        F[i] = function(sum(X[i]))
    return F


# Набор базисных функций
def basic_functions():
    func = lambda x: [(lambda i: x ** i)(i) for i in range(10)]
    return func


def pattern(train_len, test_len, t, X):
    train = np.zeros((train_len, len(x[0])))
    test = np.zeros((test_len, len(x[0])))
    t_train = np.zeros(train_len)
    t_test = np.zeros(test_len)
    indexes = []

    for i in range(train_len):
        while True:
            index = random.randint(0, N - 1)
            if len(indexes) == 0:
                indexes.append(index)
            else:
                for ind in indexes:
                    if ind == indexes:
                        continue
            for j in range(len(x[0])):
                train[i, j] = X[index, j]
            t_train[i] = t[index]
            indexes.append(index)
            break

    for i in range(test_len):
        while True:
            index = random.randint(0, N - 1)
            for ind in indexes:
                if ind == indexes:
                    continue
            for j in range(len(x[0])):
                test[i, j] = X[index, j]
            t_test[i] = t[index]
            indexes.append(index)
            break
    return train, test, t_train, t_test


def normalization(train, test, t_train, t_test):
    for j in range(len(train[0])):
        for i in range(len(train)):
            train[i, j] = (train[i, j] - train[:, j].min()) / (train[:, j].max() - train[:, j].min())

    for j in range(len(test[0])):
        for i in range(len(test)):
            test[i, j] = (test[i, j] - test[:, j].min()) / (test[:, j].max() - test[:, j].min())

    for i in range(len(t_train)):
        t_train[i] = (t_train[i] - t_train.min()) / (t_train.max() - t_train.min())

    for i in range(len(t_test)):
        t_test[i] = (t_test[i] - t_test.min()) / (t_test.max() - t_test.min())

    return train, test, t_train, t_test


# Берём x,t из датасета
x, t = fetch_california_housing(return_X_y=True)

# Коэффициент регуляризации
Lambda = 10e-10

# Колличество строк в датасете
N = len(x)

# Длины выборок
train = int(N * 0.9)  # 90%
test = int(N * 0.1)  # 10%

# Выборки
Train, Test, T_Train, T_Test = pattern(train, test, t, x)
Train, Test, T_Train, T_Test = normalization(Train, Test, T_Train, T_Test)

# Шаг
learning_rate = 0.00001

# Ограничение
eps = 0.0001

#F = design_matrix(basic_functions(), Train)
F = Train

# Начальное приближение
w0 = np.random.normal(0, 0.1, len(F[0]))

y = F @ w0
gradE = -(T_Train.T @ F).T + ((F @ w0).T @ F).T + Lambda * np.dot(w0, w0)
w1 = w0 - learning_rate * gradE

e = []
i1 = []

# Обучение
for step in range(10000):
    if np.linalg.norm(w1 - w0) < eps or np.linalg.norm(gradE) < eps:
        print("всё")
        break
    else:
        w0 = w1
        y = F @ w0.T
        gradE = -(np.dot(T_Train.T, F)).T + (w0.T @ (F.T @ F)).T + Lambda * np.dot(w1, w1)
        w1 = w0 - learning_rate * gradE
        err1 = 1/(len(F)) * (np.sum(np.power((T_Train - y), 2)) + Lambda * np.sum(np.dot(w1, w1)))
        #print(err1)
        #   мprint(gradE)
        e.append(err1)
        i1.append(step)

F = Test
y = F @ w1.T
err2 = 1/len(F) * (np.sum(np.power((T_Test - y), 2)) + Lambda * np.sum(np.dot(w0, w0)))
print(err1, err2)

plt.plot(i1, e, c="black")
plt.show()

