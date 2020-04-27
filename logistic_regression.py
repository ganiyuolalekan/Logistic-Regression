########################################################################################
# Code By Ganiyu Olalekan
########################################################################################

import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ != '__main__':

    class LogisticRegression:
        def __init__(self, x, y, alpha=0.01, num_iter=1000, verbose=False, lambd=0.0):
            self.X = x
            self.y = y
            self.lambd = lambd
            self.alpha = alpha
            self.verbose = verbose
            self.num_iter = num_iter
            self.train_x = self.train_y = self.test_x = self.test_y = self.theta = None

        # Public Methods

        def split_data(self, train_size=0.8, test_size=0.2, shuffle=True):
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
                self.X, self.y, train_size=train_size, test_size=test_size, shuffle=shuffle
            )

        def predict(self):
            return self.__predict_probability(self.test_x).round()

        def accuracy(self):
            bool_elem = (self.predict() == self.test_y)
            elem = bool_elem.size
            return (100 / elem) * bool_elem.ravel().tolist().count(True)

        def plot_boundary(self, label_1='group 1', label_2='group 2'):
            plt.figure(figsize=(10, 6))
            plt.scatter(self.X[:50, :1], self.X[:50, 1:2], color='b', label=label_1)
            plt.scatter(self.X[50:100, :1], self.X[50:100, 1:2], color='r', label=label_2)
            plt.legend()

            x1_min, x1_max = self.X[:, 0].min(), self.X[:, 0].max()
            x2_min, x2_max = self.X[:, 1].min(), self.X[:, 1].max()
            xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

            grid = np.c_[xx1.ravel(), xx2.ravel()]
            probs = self.__predict_probability(grid).reshape(xx1.shape)

            plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black')
            plt.show()

        def fit(self, timeit=True, count_at=100):
            start = time()
            count = count_at
            self.train_x = self.__add_intercept(self.train_x)

            # Initializing weights with zeros
            self.theta = np.zeros((self.train_x.shape[1], 1))

            for _ in range(self.num_iter):
                h = self.__hypothesis()
                cost = self.__cost(h)

                self.theta -= (self.alpha * self.__gradient_descent(h))

                if self.verbose and count_at == count:
                    count = 0
                    print(f"cost {(_ + 1)}: {cost}, theta: {self.theta}")

                count += 1

            if timeit:
                print(f"Ran in {round(time() - start, 2)}secs")

        ########################################################################################

        # Private Methods

        @staticmethod
        def __add_intercept(data):
            return np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)

        @staticmethod
        def __sigmoid(z):
            return 1 / (1 + np.exp(-z))

        def __cost(self, h):
            return (-self.train_y * np.log(h) - (1 - self.train_y) * np.log(1 - h)).mean()

        def __l2_regularization(self):
            return (self.lambd / 2) * np.sum(np.square(self.theta)).mean()

        def __gradient_descent(self, h):
            return (np.dot(self.train_x.T, (h - self.train_y)) / self.train_y.shape[0]) + self.__l2_regularization()

        def __hypothesis(self):
            return self.__sigmoid(np.dot(self.train_x, self.theta))

        def __predict_probability(self, data):
            return self.__sigmoid(np.dot(self.__add_intercept(data), self.theta))

        ########################################################################################


# Through Wisdom is a house built,
# by Understanding it is established and
# by Knowledge all corners and rooms are
# filled with all manner of pleasant riches
# and treasures
#
# Ref Proverbs 3: 19 - 20
