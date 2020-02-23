import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit as tt
import seaborn as sns
import csv


# # #####################################################################################################
# # #                                          Load Data                                                #
# # #####################################################################################################


def showInfo_wine(data):
    print('#######################################################################################################')
    print('#                                         Load Data                                                    ')
    print('#######################################################################################################')

    print("This is the", self.dataName, "dataset")
    print("There are", len(self.data), "set of instances")
    print("There are", len(self.data.columns), "numbers of attributes")
    print('#######################################################################################################')

    # show numbers of instances, mean, standard deviation, min, max of each feature
    pd.set_option('display.max_columns', None)

    print(data.describe())
    print('#######################################################################################################')


def showDistribution_wine(data):
    sns.distplot(data['fixed acidity'])
    plt.show()

    sns.distplot(data['volatile acidity'])
    plt.show()

    sns.distplot(data['citric acid'])
    plt.show()
    sns.distplot(data['residual sugar'])
    plt.show()
    sns.distplot(data['chlorides'])
    plt.show()
    sns.distplot(data['free sulfur dioxide'])
    plt.show()
    sns.distplot(data['total sulfur dioxide'])
    plt.show()
    sns.distplot(data['density'])
    plt.show()
    sns.distplot(data['pH'])
    plt.show()
    sns.distplot(data['sulphates'])
    plt.show()
    sns.distplot(data['alcohol'])
    plt.show()
    sns.distplot(data['quality'])
    plt.show()


def load_wine(path):
    #####################################################################################################
    #                                          Split Input/Output Variables                                         #
    #####################################################################################################

    with open(path, 'r') as f:
        wines = list(csv.reader(f, delimiter=';'))

    for items in wines[1:]:
        for quality in items[-1]:
            if quality == '5' or quality == '4' or quality == '3' or quality == '2' or quality == '1' or quality == '0':
                items[-1] = '0'
            else:
                items[-1] = '1'
    wines = np.array(wines[1:], dtype=np.float)

    minimum = wines.min(axis=0)
    maximum = wines.max(axis=0)

    for i in range(len(wines)):
        for j in range(len(wines[0]) - 1):
            wines[i][j] = (wines[i][j] - minimum[j]) / (maximum[j] - minimum[j])

    X = np.delete(wines, -1, axis=1)
    Y = wines[:, -1]
    data_train, data_test = wines[:int(len(wines) * 0.9)], wines[int(len(wines) * 0.9):]
    X_train, X_test = X[:int(len(X) * 0.9)], X[int(len(X) * 0.9):]
    Y_train, Y_test = Y[:int(len(Y) * 0.9)].T, Y[int(len(Y) * 0.9):].T

    return data_train, data_test, X_train, Y_train, X_test, Y_test


def load2_0_wine(path):
    #####################################################################################################
    #                                          Feature selection                                        #
    #####################################################################################################

    with open(path, 'r') as f:
        wines = list(csv.reader(f, delimiter=';'))

    for items in wines[1:]:
        for quality in items[-1]:
            if quality == '5' or quality == '4' or quality == '3' or quality == '2' or quality == '1' or quality == '0':
                items[-1] = '0'
            else:
                items[-1] = '1'
    wines = np.array(wines[1:], dtype=np.float)

    X = np.delete(wines, -1, axis=1)
    Y = wines[:, -1]

    add1 = np.empty(len(X))
    add2 = np.empty(len(X))
    add3 = np.empty(len(X))
    add4 = np.empty(len(X))
    for k in range(len(X)):
        add1[k] = X[k, 1] * X[k, 7]
        add2[k] = X[k, 5] * X[k, 8]
        add3[k] = X[k, 3] * X[k, 10]
        add4[k] = X[k, 0] * X[k, 0]
    add6 = np.ones(len(X))

    X = np.insert(X, 11, values=add1, axis=1)
    X = np.insert(X, 12, values=add2, axis=1)
    X = np.insert(X, 13, values=add3, axis=1)
    # X = np.insert(X,14,values=add4,axis=1)
    # X = np.insert(X,0,values=add6,axis=1)
    # X = np.delete(X,5,axis=1)

    Y = Y.reshape(np.size(Y), 1)

    wine_new = np.concatenate((X, Y), axis=1)

    minimum = wine_new.min(axis=0)
    maximum = wine_new.max(axis=0)

    for i in range(len(wine_new)):
        for j in range(len(wine_new[0]) - 1):
            wine_new[i][j] = (wine_new[i][j] - minimum[j]) / (maximum[j] - minimum[j])

    X = np.delete(wine_new, -1, axis=1)
    Y = wine_new[:, -1]

    data_train, data_test = wine_new[:int(len(wine_new) * 0.9)], wines[int(len(wine_new) * 0.9):]
    X_train, X_test = X[:int(len(X) * 0.9)], X[int(len(X) * 0.9):]
    Y_train, Y_test = Y[:int(len(Y) * 0.9)].T, Y[int(len(Y) * 0.9):].T

    # data = {"data_train", data_train,
    #         "data_test", data_test,
    #         "X_train", X_train,
    #         "Y_train", Y_train,
    #         "X_test", X_test,
    #         "Y_test", Y_test}
    return data_train, data_test, X_train, Y_train, X_test, Y_test


def showInfo(data):
    print("Not yet implemented")


def showDistribution(data):
    print("note yet implemented")


def loadCancer(path):
    f = open(path, 'r')
    df = pd.DataFrame(list(f))
    drop_index = []
    data = np.zeros(shape=[1, 11])

    #####################################################################################################
    #                                           deleting non-numerical value                                              #
    #####################################################################################################

    for index, row in df.iterrows():
        strings = row.to_string()
        strings = strings.replace('\\n', '')
        strings = strings.replace('0    ', '')
        value = strings.split(',')

        for i in range(len(value)):
            if value[i].isdigit() == 0:
                drop_index.append(index)

    df_new = df.drop(index=drop_index)

    for indexs, rows in df_new.iterrows():
        string = rows.to_string()
        string = string.replace('\\n', '')
        string = string.replace('0    ', '')
        values = string.split(',')
        onedata = np.array(values, dtype='float64')
        onedata = onedata.reshape(1, len(onedata))
        data = np.append(data, onedata, axis=0)

    data = np.delete(data[1:], 0, axis=1)
    X = np.delete(data, -1, axis=1)
    # add = np.ones(len(X))
    # X = np.insert(X, 0, values=add, axis=1)
    Y = data[:, -1]

    # =============================================================================
    # Normalization
    # =============================================================================
    for p in range(len(X)):
        for q in range(len(X[0])):
            X[p][q] = X[p][q] / 1

    # =============================================================================
    # define classes and split data set
    # =============================================================================
    for x in range(np.size(Y)):
        if Y[x] == 2:
            Y[x] = 0
        else:
            Y[x] = 1

    data_train, data_test = data[:int(len(data) * 0.9)], data[int(len(data) * 0.9):]
    X_train, X_test = X[:int(len(X) * 0.9)], X[int(len(X) * 0.9):]
    Y_train, Y_test = Y[:int(len(Y) * 0.9)].T, Y[int(len(Y) * 0.9):].T

    return data_train, data_test, X_train, Y_train, X_test, Y_test


#####################################################################################################
#                                          Main Functions                                              #
#####################################################################################################


def load(path, name):
    if name == "wine_quality":
        # self.data = loadWine.load(path)
        # data = pd.read_csv(path, delimiter=';', encoding='utf-8')
        # loadWine.showInfo(self.data)
        return load2_0_wine(path)

    elif name == "breast_cancer":
        # loadCancer.showInfo(self.data)

        return loadCancer(path)
    else:
        raise "Invalid Dataset. Your options are 'wine_quality' or 'breast_cancer'"


# def showDistribution(name):
#     if name == "wine_quality":
#         loadWine.showDistribution(self.data)
#     elif name == "breast_cancer":
#         loadCancer.showDistribution(self.data)

# def showInfo():
#     if self.name == "wine_quality":
#         loadWine.showInfo(self.data)
#     elif self.name == "breast_cancer":
#         loadCancer.showInfo(self.data)

def split(data_train):
    batch1 = data_train[0:int(len(data_train) / 5) + 1, :]
    batch2 = data_train[int(len(data_train) / 5) + 1:int(len(data_train) * 2 / 5) + 1, :]
    batch3 = data_train[int(len(data_train) * 2 / 5) + 1:int(len(data_train) * 3 / 5) + 1, :]
    batch4 = data_train[int(len(data_train) * 3 / 5) + 1:int(len(data_train) * 4 / 5) + 1, :]
    batch5 = data_train[int(len(data_train) * 4 / 5) + 1:, :]

    train_1 = np.concatenate((batch1, batch2, batch3, batch4))
    valid_1 = batch5

    train_2 = np.concatenate((batch1, batch2, batch3, batch5))
    valid_2 = batch4

    train_3 = np.concatenate((batch1, batch2, batch4, batch5))
    valid_3 = batch3

    train_4 = np.concatenate((batch1, batch3, batch4, batch5))
    valid_4 = batch2

    train_5 = np.concatenate((batch2, batch3, batch4, batch5))
    valid_5 = batch1

    train = {"train_1": train_1,
             "train_2": train_2,
             "train_3": train_3,
             "train_4": train_4,
             "train_5": train_5,
             }

    valid = {"valid_1": valid_1,
             "valid_2": valid_2,
             "valid_3": valid_3,
             "valid_4": valid_4,
             "valid_5": valid_5,
             }

    return train_1, train_2, train_3, train_4, train_5, valid_1, valid_2, valid_3, valid_4, valid_5