import numpy as np
import timeit as tt
from data import loadData
from model.log_reg import logistic_regression

# # #####################################################################################################
# # #                                          Load Data                                                #
# # #####################################################################################################


#r before the path is used to load the file in pycharm
#if you use other IDEs, it will arise error!!

#r before the path is used to load the file in pycharm
#if you use other IDEs, it will arise error!!

#r before the path is used to load the file in pycharm
#if you use other IDEs, it will arise error!!

# data_train, data_test, X_train, Y_train, X_test, Y_test = loadData.load(
#     r'C:\Users\Gary\PycharmProjects\COMP551MINI1\Final\FINAL!!\data\data\winequality-red.csv', 'wine_quality')
data_train, data_test, X_train,Y_train, X_test, Y_test = loadData.load(
    r'C:\Users\Gary\PycharmProjects\COMP551MINI1\Final\FINAL!!\data\data\breast-cancer-wisconsin.data', 'breast_cancer')

# print(data_train.shape)
# print(data_test.shape)
# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)


train_1, train_2, train_3, train_4, train_5, valid_1, valid_2, valid_3, valid_4, valid_5 = loadData.split(data_train)
# print(train_1.shape)
# print(train_2.shape)
# print(train_3.shape)
# print(train_4.shape)
# print(train_5.shape)


# print(valid_1.shape)
# print(valid_2.shape)
# print(valid_3.shape)
# print(valid_4.shape)
# print(valid_5.shape)


# hyperparameters
accuracy = np.zeros(5)
train_accuracy = np.zeros(5)
learning_rate = 0.01
I_time = 1000


# helper function
def split_xy(v):
    split_x = np.delete(v, -1, axis=1)
    split_y = v[:, -1]
    return split_x, split_y


def train(x_train, y_train, x_valid, y_valid):
    model = logistic_regression(x_train, y_train, learning_rate, I_time)
    weight_train = model.fit()
    y_pred = model.predict(x_train, weight_train)
    train_accuracy = model.evaluate_acc(y_pred, y_train)

    # #validation
    weight = model.fit()
    y_pred_valid = model.predict(x_valid, weight)
    valid_accuracy = model.evaluate_acc(y_pred_valid, y_valid)

    return weight_train,train_accuracy, valid_accuracy


#####################################################################################################
#                                          training with 5-cross validation                                             #
#####################################################################################################
tot_start = tt.default_timer()

print("==============================================================")
print("k = 1")
print("==============================================================")
x_train, y_train = split_xy(train_1)
x_valid, y_valid = split_xy(valid_1)

start = tt.default_timer()
w1,train_accuracy[0], accuracy[0] = train(x_train, y_train, x_valid, y_valid)
stop = tt.default_timer()

print('Running time:', stop - start)
print('Accuracy:', train_accuracy[0])

print("==============================================================")
print("k = 2")
print("==============================================================")
x_train, y_train = split_xy(train_2)
x_valid, y_valid = split_xy(valid_2)

start = tt.default_timer()
w2,train_accuracy[1], accuracy[1] = train(x_train, y_train, x_valid, y_valid)
stop = tt.default_timer()

print('Running time:', stop - start, 's')
print('Accuracy:', train_accuracy[1])

print("==============================================================")
print("k = 3")
print("==============================================================")
x_train, y_train = split_xy(train_3)
x_valid, y_valid = split_xy(valid_3)

start = tt.default_timer()
w3,train_accuracy[2], accuracy[2] = train(x_train, y_train, x_valid, y_valid)
stop = tt.default_timer()

print('Running time:', stop - start, 's')
print('Accuracy:', train_accuracy[2])

print("==============================================================")
print("k = 4")
print("==============================================================")
x_train, y_train = split_xy(train_4)
x_valid, y_valid = split_xy(valid_4)

start = tt.default_timer()
w4,train_accuracy[3], accuracy[3] = train(x_train, y_train, x_valid, y_valid)
stop = tt.default_timer()

print('Running time:', stop - start, 's')
print('Accuracy:', train_accuracy[3])

print("==============================================================")
print("k = 5")
print("==============================================================")
x_train, y_train = split_xy(train_5)
x_valid, y_valid = split_xy(valid_5)

start = tt.default_timer()
w5,train_accuracy[4], accuracy[4] = train(x_train, y_train, x_valid, y_valid)
stop = tt.default_timer()

print('Running time:', stop - start, 's')
print('Accuracy:', train_accuracy[4])

#####################################################################################################
#                                          Test                                            #
#####################################################################################################

tot_stop = tt.default_timer()
print("==============================================================")
print("total runtime:", tot_stop - tot_start)
print("==============================================================")

mean_accuracy_train = np.mean(train_accuracy, 0)

print('mean training accuracy:', mean_accuracy_train, '%')
mean_accuracy = np.mean(accuracy, 0)

print('mean validation accuracy:', mean_accuracy, '%')

model = logistic_regression(X_test, Y_test, learning_rate, I_time)
# weight = model.fit()
y_pred = model.predict(X_test, w5)
accuracy = model.evaluate_acc(y_pred, Y_test)

print("Test accuracy:", accuracy)


