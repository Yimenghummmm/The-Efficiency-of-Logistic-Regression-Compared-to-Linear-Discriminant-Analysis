import numpy as np
import csv
import math
from numpy.linalg import inv


class LDA():
    def __init__(self, x, y, x_1, x_0, y_1, y_0):
        self.x = x  # input x
        self.y = y  # input y
        self.x_1 = x_1  # class 1 x
        self.x_0 = x_0  # class 0 x
        self.y_1 = y_1  # class 1 y
        self.y_0 = y_0  # class 0 y

    def dot3(self, x, y, z):
        w = np.dot(y, z)
        return np.dot(x, w)

    def decision_boundary(self, x):
        for i in range(np.size(x, 0)):
            if int(x[i]) > 0:
                x[i] = 1
            else:
                x[i] = 0
        return x

    def fit(self):

        # calculation of P(y=1) and P(y=0)
        P_y1 = np.size(self.y_1) / np.size(self.y)
        P_y0 = 1 - P_y1  # two numbers

        # calculate the average vector u0 and u1
        u0 = np.mean(self.x_0, 0).T
        u1 = np.mean(self.x_1, 0).T  # two row vectors, but in one dimension

        # calculate the covariance
        u0_u1_cov = np.zeros([np.size(self.x, 1), np.size(self.x, 1)])  # an m by m vector

        for i in range(np.size(self.y, 0)):

            if self.y[i] == 0:
                x_mid = self.x[i, :]
                x_midC = x_mid.reshape(np.size(self.x, 1), 1)
                x_midR = x_midC.T
                # print(np.dot(x_midC,x_midR))
                u0_u1_cov += np.dot(x_midC, x_midR)
                # print(u0_u1_cov)
            elif self.y[i] == 1:
                x_mid = self.x[i, :]
                x_midC = x_mid.reshape(np.size(self.x, 1), 1)
                x_midR = x_midC.T
                # print(np.dot(x_midC, x_midR))
                u0_u1_cov += np.dot(x_midC, x_midR)
                # print(u0_u1_cov)
            else:
                print('invalid' + self.y[i] + 'in position' + i)
                raise NameError('invalid data in y')

        u0_u1_cov = u0_u1_cov / (np.size(self.y) - 2)
        # calculate the log_odd_ratio
        log_odd_ratio = np.zeros([np.size(self.y, 0), 1])  # an n by 1 vector

        u0 = u0.reshape(np.size(self.x, 1), 1)
        u1 = u1.reshape(np.size(self.x, 1), 1)

        log1 = math.log(P_y1 / P_y0, 10)
        log2 = 0.5 * self.dot3(u1.T, inv(u0_u1_cov), u1)
        log3 = 0.5 * self.dot3(u0.T, inv(u0_u1_cov), u0)
        log4 = np.dot(inv(u0_u1_cov), u1 - u0)

        # for j in range(np.size(self.y, 0)):
        #
        #     x_T = self.x[j, :].reshape(1, np.size(self.x, 1))
        #
        #
        #     log_odd_ratio[j] = log1 - log2 + log3 + self.dot3(x_T, inv(u0_u1_cov), u1 - u0)
        #
        #     # log_odd_ratio[j] = math.log(P_y1/P_y0,10) - 0.5 * self.dot3(u1.T,inv(u0_u1_cov),u1) + \
        #     #                 0.5 * self.dot3(u0.T,inv(u0_u1_cov),u0) + \
        #     #                 self.dot3(self.x[i,:].reshape(1,np.size(self.x, 1)),inv(u0_u1_cov),u1 - u0)
        #
        # # print(log_odd_ratio)

        return log1, log2, log3, log4

    def predict(self, x_train, log1, log2, log3, log4):

        # log1, log2, log3, log4 = self.fit()

        log_odd_ratio = np.zeros([np.size(x_train, 0), 1])  # an n by 1 vector

        for l in range(np.size(x_train, 0)):
            x_T = x_train[l, :].reshape(1, np.size(self.x, 1))

            log_odd_ratio[l] = log1 - log2 + log3 + np.dot(x_T, log4)

        return self.decision_boundary(log_odd_ratio)

    def evaluate_acc(self, y_predict, y_test):

        log_odd_ratio = y_predict

        number_of_all_sets = np.size(y_test, 0)

        y_test = y_test.reshape(np.size(y_test, 0), 1)

        number_of_false_predict = np.sum(abs(np.subtract(y_test, log_odd_ratio)))

        accuracy = 1 - number_of_false_predict / number_of_all_sets

        print('accuracy is', float(accuracy) * 100, '%')

        return float(accuracy) * 100