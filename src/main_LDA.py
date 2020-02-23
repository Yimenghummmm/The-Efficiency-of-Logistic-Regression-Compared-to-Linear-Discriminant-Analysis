import os
import sys
scriptpath = "c:/Users/yimen/Desktop/comp_551/A1"
# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(scriptpath))

import numpy as np
import timeit as tt
from data import loadData

from model.LDA import LDA

#r before the path is used to load the file in pycharm
#if you use other IDEs, it will arise error!!

#r before the path is used to load the file in pycharm
#if you use other IDEs, it will arise error!!

#r before the path is used to load the file in pycharm
#if you use other IDEs, it will arise error!!

data_train, data_test, X_train,Y_train, X_test, Y_test = loadData.load(r'C:\Users\Gary\PycharmProjects\COMP551MINI1\Final\FINAL!!\data\data\winequality-red.csv', 'wine_quality')
# data_train, data_test, X_train,Y_train, X_test, Y_test = loadData.load(r'C:\Users\Gary\PycharmProjects\COMP551MINI1\Final\FINAL!!\data\data\breast-cancer-wisconsin.data', 'breast_cancer')

train_1,train_2,train_3,train_4,train_5, valid_1,valid_2,valid_3,valid_4,valid_5 = loadData.split(data_train)

#helper function
def split_xy(v):
    split_x = np.delete(v, -1, axis=1)
    split_y = v[:, -1]
    return split_x, split_y

def class_sort(d):
    d_neg = np.zeros(shape=[1, len(d[0])])
    d_pos = np.zeros(shape=[1, len(d[0])])
    for j in range(len(d)):
        if d[j, -1] == 0:
            d_neg = np.append(d_neg, d[j].reshape(1, len(d[j])), axis=0)
        else:
            d_pos = np.append(d_pos, d[j].reshape(1, len(d[j])), axis=0)
    d_neg = np.delete(d_neg, 0, axis=0)
    d_pos = np.delete(d_pos, 0, axis=0)
    x_train_neg = np.delete(d_neg, -1, axis=1)
    x_train_pos = np.delete(d_pos, -1, axis=1)
    y_train_neg = d_neg[:, -1]
    y_train_pos = d_pos[:, -1]
    return x_train_neg, x_train_pos, y_train_neg, y_train_pos



#####################################################################################################
#                                          training with 5-cross validation                         #
#####################################################################################################

accuracy = np.zeros(5)
train_accuracy = np.zeros(5)

tot_start = tt.default_timer()

print("==============================================================")
print("k = 1")
print("==============================================================")

start = tt.default_timer()

train_xneg, train_xpos, train_yneg, train_ypos = class_sort(train_1)
x_train, y_train = split_xy(train_1)
x_valid, y_valid = split_xy(valid_1)
model = LDA(x_train,y_train, train_xpos, train_xneg,train_ypos, train_yneg)

log1, log2, log3, log4 = model.fit()
y_pred = model.predict(x_train, log1, log2, log3, log4)

train_accuracy[0] = model.evaluate_acc(y_pred, y_train)
y_pred_valid = model.predict(x_valid, log1, log2, log3,log4)
accuracy[0] = model.evaluate_acc(y_pred_valid, y_valid)

stop = tt.default_timer()

print("Accuracy:" ,accuracy[0])
print("Time taken:", stop-start)


print("==============================================================")
print("k = 2")
print("==============================================================")

start = tt.default_timer()

train_xneg, train_xpos, train_yneg, train_ypos = class_sort(train_2)
x_train, y_train = split_xy(train_2)
x_valid, y_valid = split_xy(valid_2)
model = LDA(x_train,y_train, train_xpos, train_xneg,train_ypos, train_yneg)
model2 = LDA(x_train,y_train, train_xpos, train_xneg,train_ypos, train_yneg)

log1, log2, log3, log4 = model.fit()
y_pred = model.predict(x_train, log1, log2, log3, log4)

train_accuracy[1] = model.evaluate_acc(y_pred, y_train)
y_pred_valid = model.predict(x_valid, log1, log2, log3,log4)
accuracy[1] = model.evaluate_acc(y_pred_valid, y_valid)

stop = tt.default_timer()

print("Accuracy:" ,accuracy[1])
print("Time taken:", stop-start)



print("==============================================================")
print("k = 3")
print("==============================================================")

start = tt.default_timer()

train_xneg, train_xpos, train_yneg, train_ypos = class_sort(train_3)
x_train, y_train = split_xy(train_3)
x_valid, y_valid = split_xy(valid_3)
model = LDA(x_train,y_train, train_xpos, train_xneg,train_ypos, train_yneg)

log1, log2, log3, log4 = model.fit()
y_pred = model.predict(x_train, log1, log2, log3, log4)

train_accuracy[2] = model.evaluate_acc(y_pred, y_train)
y_pred_valid = model.predict(x_valid, log1, log2, log3,log4)
accuracy[2] = model.evaluate_acc(y_pred_valid, y_valid)

stop = tt.default_timer()

print("Accuracy:" ,accuracy[2])
print("Time taken:", stop-start)



print("==============================================================")
print("k = 4")
print("==============================================================")

start = tt.default_timer()

train_xneg, train_xpos, train_yneg, train_ypos = class_sort(train_4)
x_train, y_train = split_xy(train_3)
x_valid, y_valid = split_xy(valid_3)
model = LDA(x_train,y_train, train_xpos, train_xneg,train_ypos, train_yneg)
model4 = LDA(x_train,y_train, train_xpos, train_xneg,train_ypos, train_yneg)

log1, log2, log3, log4 = model.fit()
y_pred = model.predict(x_train, log1, log2, log3, log4)

train_accuracy[3] = model.evaluate_acc(y_pred, y_train)
y_pred_valid = model.predict(x_valid, log1, log2, log3,log4)
accuracy[3] = model.evaluate_acc(y_pred_valid, y_valid)

stop = tt.default_timer()

print("Accuracy:" ,accuracy[3])
print("Time taken:", stop-start)



print("==============================================================")
print("k = 5")
print("==============================================================")

start = tt.default_timer()

train_xneg, train_xpos, train_yneg, train_ypos = class_sort(train_5)
x_train, y_train = split_xy(train_5)
x_valid, y_valid = split_xy(valid_5)
model = LDA(x_train,y_train, train_xpos, train_xneg,train_ypos, train_yneg)

log1, log2, log3, log4 = model.fit()
y_pred = model.predict(x_train, log1, log2, log3, log4)

train_accuracy[4] = model.evaluate_acc(y_pred, y_train)
y_pred_valid = model.predict(x_valid, log1, log2, log3,log4)
accuracy[4] = model.evaluate_acc(y_pred_valid, y_valid)

stop = tt.default_timer()

print("Accuracy:" ,accuracy[4])
print("Time taken:", stop-start)



# # #####################################################################################################
# # #                                          Test Evaluation                                               #
# # #####################################################################################################
tot_stop = tt.default_timer()
print("==============================================================")
print("total time:", tot_stop - tot_start)
print("==============================================================")

mean_accuracy_train = np.mean(train_accuracy, 0)
print('mean train accuracy is', mean_accuracy_train, '%')


mean_accuracy = np.mean(accuracy, 0)
print('mean validation accuracy is', mean_accuracy, '%')


# # test set#####################################################################################################

log1, log2, log3, log4 = model2.fit()

y_predict_test = model2.predict(X_test,log1, log2, log3, log4)

accuracy_test = model2.evaluate_acc(y_predict_test,Y_test)