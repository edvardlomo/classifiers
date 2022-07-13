import numpy as np
from sklearn.datasets import make_blobs
rnd_seed = 2022

def get_data(datapoints, cs):
    """Gets all data. The classes are given with centers cs, where each center corresponds to one class."""

    return make_blobs(n_samples=[datapoints//cs.shape[0]]*cs.shape[0], centers=cs,
            n_features=cs.shape[1], random_state=rnd_seed)

def split_data(X, t):
    """Splits data into a train, val, test set"""

    # Gets indices of elements to include different classes in the sets
    indices = np.arange(X.shape[0])
    rng = np.random.RandomState(rnd_seed)
    rng.shuffle(indices)

    # Split set into subsets
    train_const = X.shape[0]//2
    val_const = (X.shape[0]*3)//4
    X_train = X[indices[:train_const],:]
    X_val = X[indices[train_const:val_const],:]
    X_test = X[indices[val_const:],:]
    t_train = t[indices[:train_const]]
    t_val = t[indices[train_const:val_const]]
    t_test = t[indices[val_const:]]
    return (X_train, X_val, X_test), (t_train, t_val, t_test)
