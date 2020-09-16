import numpy as np
from sklearn.model_selection import train_test_split

def loadData(file_d, file_l):
    y = np.genfromtxt(file_l, delimiter=",", dtype="int")
    X = np.genfromtxt(file_d, delimiter=",", dtype="float")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test

