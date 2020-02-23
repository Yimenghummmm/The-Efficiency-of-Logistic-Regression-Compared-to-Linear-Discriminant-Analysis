import os
import sys

scriptpath = "c:/Users/yimen/Desktop/comp_551/A1"
# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(scriptpath))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# np.random.seed(0)
class logistic_regression():
    def __init__(self, x, y, learning_rate, iteration_time, ):
        self.x = x  # input x
        self.y = y.reshape(np.size(y, 0), 1)  # input y
        self.learning_rate = learning_rate  # learning rate
        self.iteration_time = iteration_time  # iteration times

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def decision_boundary(self, x):
        for i in range(np.size(x, 0)):
            if int(x[i]) > 0:
                x[i] = 1
            else:
                x[i] = 0
        return x

    def fit(self):

        np.random.seed(0)

        w = np.random.rand(np.size(self.x, 1), 1)

        # print('random weight is', w)

        for p in range(self.iteration_time):

            log_likelihood = 0

            gradient = np.zeros([np.size(self.x, 1), 1])

            for i in range(np.size(self.y, 0)):

                x_mid = self.x[i, :]
                x_midC = x_mid.reshape(np.size(self.x, 1), 1)
                x_midR = x_midC.T

                y_pred = np.dot(w.T, x_midC)
                y_pred = self.sigmoid(y_pred)  # should be a number

                if int(y_pred) >= 1:
                    y_pred -= 0.02
                # print('too big value in ypred', i)

                if int(self.y[i]) == 1:

                    log_likelihood += np.log(y_pred)

                elif int(self.y[i]) == 0:
                    log_likelihood += np.log(1 - y_pred)

                else:
                    print('invalid' + self.y[i] + 'in position' + i)
                    raise NameError('invalid data in y')

                gradient += (self.y[i] - y_pred) * x_midC

            cross_entropy = - log_likelihood

            w = w + self.learning_rate * gradient
            # print('cross entropy is ', cross_entropy)
            #
            # print('iteration number is ', p)

        # print('Loss:', cross_entropy)  # a number

        # print('final w is', w) # an n by 1 vector

        return w

    def fit_adaptive(self):

        np.random.seed(0)

        w = np.random.rand(np.size(self.x, 1), 1)

        # print('random weight is', w)

        p = 1

        gradient_1 = np.zeros([np.size(self.x, 1), 1])

        gradient_2 = np.zeros([np.size(self.x, 1), 1])

        gradient_3 = np.zeros([np.size(self.x, 1), 1])

        q = 0;

        while (q <= 3):

            log_likelihood = 0

            gradient = np.zeros([np.size(self.x, 1), 1])

            for i in range(np.size(self.y, 0)):

                x_mid = self.x[i, :]
                x_midC = x_mid.reshape(np.size(self.x, 1), 1)
                x_midR = x_midC.T

                y_pred = np.dot(w.T, x_midC)
                y_pred = self.sigmoid(y_pred)  # should be a number

                if int(y_pred) >= 1:
                    y_pred -= 0.01
                    # print('error in ypred', i)

                if int(self.y[i]) == 1:

                    log_likelihood += np.log(y_pred)

                elif int(self.y[i]) == 0:
                    log_likelihood += np.log(1 - y_pred)

                else:
                    print('invalid' + self.y[i] + 'in position' + i)
                    raise NameError('invalid data in y')

                gradient += (self.y[i] - y_pred) * x_midC

            gradient_3 = gradient_2

            gradient_2 = gradient_1

            gradient_1 = gradient

            gradient_dif1 = np.mean(abs(gradient_1 - gradient_2))

            gradient_dif2 = np.mean(abs(gradient_2 - gradient_3))

            gradient_dif = abs(gradient_dif2 - gradient_dif1)

            if gradient_dif1 != 0 and gradient_dif2 != 0:
                gradient_percent = gradient_dif / gradient_dif2
                # print('difference is',gradient_percent)
                if gradient_percent <= 0.001:
                    q += 1

            cross_entropy = - log_likelihood

            w = w + self.learning_rate * gradient
            # print('cross entropy is ', cross_entropy)
            #
            # print('iteration number is ', p)

            p += 1

        print('last gradient difference is', gradient_percent * 100, '%')

        print('final cross entropy is', cross_entropy)  # a number

        print('final iteration number is', p)  # a number

        # print('final w is', w)  # an n by 1 vector

        return w

    def predict(self, x_train, weight):

        w = weight

        a = np.zeros([np.size(x_train, 0), 1])  # an n by 1 vector

        for l in range(np.size(x_train, 0)):
            x_T = x_train[l, :].reshape(1, np.size(self.x, 1))

            a[l] = np.dot(x_T, w)

        return self.decision_boundary(a)

    def evaluate_acc(self, y_predcit, y_test):

        number_of_all_sets = np.size(y_test, 0)
        y_test = y_test.reshape(np.size(y_test, 0), 1)
        number_of_false_predict = np.sum(abs(np.subtract(y_test, y_predcit)))

        accuracy = 1 - number_of_false_predict / number_of_all_sets

        # print('accuracy is', float(accuracy) * 100, '%')

        return float(accuracy) * 100