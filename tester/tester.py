import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import code_logistic_regression.logistic_regression as logic
import sys
from code_misc.utils import MyUtils

def main():
    (X_train,y_train,X_test,y_test) = loadData()
    passedGD = passedSGD = False
    testSigmoid()
    passedGD = testGD(X_train,y_train,X_test,y_test)
    passedSGD = testSGD(X_train,y_train,X_test,y_test)
    if passedGD and passedSGD:
        print("SUCCESSFUL RUN!")
    else:
        print(f'PassedGD: {passedGD}, PassedSGD: {passedSGD}')
        print("Non-shuffling of the data is assumed as well for the tester's data.")


def testGD(X_train,y_train,X_test,y_test):
    errors = loadErrors('GD_Error.npz')
    row = 0
    passed = True

    for lam in [0,10]:
        for z_r in [1,2]:
            for eta in [0.01,0.001]:
                log = logic.LogisticRegression() #Create a new lr object each time. No assumption made that the weights will reset.
                log.fit(X_train, y_train, lam = lam, eta = eta, iterations = 10000, SGD = False, mini_batch_size = 20, degree = z_r)
                train_error = log.error(X_train, y_train)
                test_error = log.error(X_test, y_test)

                (mikes_train_error,mikes_test_error) = errors[row]
                row+=1
                if not inThreshold(train_error,mikes_train_error) or not inThreshold(test_error,mikes_test_error):
                    print(f'For GD, and the following params:\nlam: {lam}, z_r: {z_r}, eta: {eta}')
                    print(f'Expected train/test error:\n{mikes_train_error}, {mikes_test_error}')
                    print(f'Found train/test error:\n{train_error}, {test_error}\n')
                    passed = False
    if not passed:
        print("Please note that for Gradient descent, 0 initialization of the weights is assumed.")
        print("Due to randomness resulting from shuffling the data, minor differences can be safely ignored.")
    return passed

def testSGD(X_train,y_train,X_test,y_test):
    errors = loadErrors('SGD_Error.npz')
    row = 0
    passed = True

    n,d = X_train.shape
    for lam in [0,1]:
        for z_r in [1,2]:
            for eta in [0.01,0.001]:
                for mbs in [1,20,n]:
                    log = logic.LogisticRegression() #Create a new log object each time. No assumption made that the weights will reset.
                    log.fit(X_train, y_train, lam = lam, eta = eta, iterations = 10000, SGD = True, mini_batch_size = mbs, degree = z_r)
                    train_error = log.error(X_train, y_train)
                    test_error = log.error(X_test, y_test)

                    (mikes_train_error,mikes_test_error) = errors[row]
                    row+=1
                    if not inThreshold(train_error,mikes_train_error) or not inThreshold(test_error,mikes_test_error):
                        print(f'For SGD, and the following params:\nlam: {lam}, z_r: {z_r}, eta: {eta}')
                        print(f'Expected train/test error:\n{mikes_train_error}, {mikes_test_error}')
                        print(f'Found train/test error:\n{train_error}, {test_error}\n')
                        passed = False
    if not passed:
        print("Please note that for Stochastic Gradient descent, 0 initialization of the weights is assumed.")
        print("Due to randomness resulting from shuffling the data, minor differences can be safely ignored.")
    return passed


def testSigmoid():
    actualValue = [0.5,0.7310586,0.8807971,0.2689414]
    i = 0
    for s in [0,1,2,-1]:
        assert inThreshold(logic.LogisticRegression._sigmoid(s),actualValue[i],0.00001), f"Incorrect sigmoid value, expected {actualValue[i]}, but found {logic.LogisticRegression._sigmoid(s)} for s = {s}"
        i += 1

def inThreshold(x1, x2, threshold=1.1):
    return abs(x1 - x2) < threshold

def loadData(data_set='ionoshpere'):
    #Reads the files into pandas dataframes from the respective .csv files.
    path = '../code_logistic_regression/ionosphere'
    df_X_train = pd.read_csv(f'{path}/X_train.csv', header=None)
    df_y_train = pd.read_csv(f'{path}/y_train.csv', header=None)
    df_X_test = pd.read_csv(f'{path}/X_test.csv', header=None)
    df_y_test = pd.read_csv(f'{path}/y_test.csv', header=None)

    #Convert the input data into numpy arrays and normalize.
    X_train = df_X_train.to_numpy()
    X_test = df_X_test.to_numpy()
    n_train = X_train.shape[0]

    X_all = MyUtils.normalize_neg1_pos1(np.concatenate((X_train, X_test), axis=0))
    X_train = X_all[:n_train]
    X_test = X_all[n_train:]

    y_train = df_y_train.to_numpy()
    y_test = df_y_test.to_numpy()

    #Insure that the data correctly loaded in.
    assert X_train.shape == (280, 34), "Incorrect input, expected (280, 34), found " + X_train.shape
    assert y_train.shape == (280, 1), "Incorrect input, expected (280, 1), found " + y_train.shape
    assert X_test.shape  == (71, 34), "Incorrect input, expected (71, 34), found " + X_test.shape
    assert y_test.shape  == (71, 1), "Incorrect input, expected (71, 1), found " + y_test.shape

    return (X_train,y_train,X_test,y_test)

def loadErrors(file):
    container = np.load(file)
    data = [container[key] for key in container]
    errors = np.array(data)
    return errors

if __name__ == '__main__':
    main()