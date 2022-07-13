import numpy as np


def logistic(x):
    return 1/(1+np.exp(-x))


def normalize_feature(X):
    X = X.copy()
    for i in range(X.shape[1]):
        f = X[:, i]
        f_mean = np.mean(f)
        f_std = np.std(f)
        X[:, i] = (f - f_mean)/f_std
    return X

def one_hot_encoding(t):
    """One-hot-encoding"""
    t_ = np.zeros((t.shape[0], len(set(t))))
    for i,c in enumerate(t):
        t_[i][c] = 1
    return t_

def add_bias(X):
    """Adds bias to X"""
    if len(X.shape) == 1:
        # X is a vector
        return np.concatenate([np.array([1]), X])
    else:
        # X is a matrix
        m = X.shape[0]
        bias = np.ones((m,1)) # Makes a m*1 matrix of 1-s
        return np.concatenate([bias, X], axis=1) 

# Scores
def confusion_matrix(model, X, t):
    features = len(set(t))
    conmatrix = np.zeros((features,features)).astype("int")
    t_pred = model.predict(X).astype("int")

    # Gold standard along columns
    for i,j in zip(t_pred, t):
        conmatrix[i][j] += 1
    return conmatrix

def accuracy(conmatrix, *args):
    if conmatrix == None:
        conmatrix = confusion_matrix(*args)
    return conmatrix.diagonal().sum()/conmatrix.sum()

def precision(conmatrix, c, *args):
    if conmatrix == None:
        conmatrix = confusion_matrix(*args)
    column = conmatrix[:,c]
    return column[c]/column.sum()

def recall(conmatrix, r, *args):
    if conmatrix == None:
        conmatrix = confusion_matrix(*args)
    row = conmatrix[r]
    return row[r]/row.sum()
