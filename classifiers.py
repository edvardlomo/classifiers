import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
rnd_seed = 2022


class LinRegModel():
    
    def fit(self, X, t, eta=0.1, epochs=10):
        """X is a Nxm matrix, where N is datapoints and m features, t is target values for X"""

        k, m = X.shape
        X_biased = add_bias(X)

        self.ws = np.zeros(m + 1)
        for e in range(epochs):
            self.ws -= eta / k * X_biased.T @ (X_biased @ self.ws - t)

    def predict(self, X, threshold=0.5):
        z = add_bias(X)
        score = z @ self.ws
        return score>threshold


def logistic(x):
    return 1/(1+np.exp(-x))

class LogRegModel():
    
    def fit(self, X, t, X_val=None, t_val=None, eta = 0.1, epochs=10, loss_diff=0):
        """X is a Nxm matrix, where N is datapoints and m features, t is target values for X"""
        
        has_val_set = type(X_val) in [list, np.ndarray]
        (k, m) = X.shape
        X_biased = add_bias(X)
        
        self.ws = np.zeros(m+1)
        accs = []
        losses = []
        for e in range(epochs):
            self.ran_epochs = e
            t_pred = self.forward(X_biased)
            self.ws -= eta / k *  X_biased.T @ (t_pred - t)
            l = -np.sum(t *  np.log(t_pred))/len(t_pred)
            losses.append(l)
            if has_val_set:
                accs.append((self.accuracy(X, t), self.accuracy(X_val, t_val)))
        return losses,accs
    
    def forward(self, X):
        return logistic(X @ self.ws)
    
    def score(self, X):
        z = add_bias(X)
        score = self.forward(z)
        return score
    
    def predict(self, X, threshold=0.5):
        z = add_bias(X)
        score = self.forward(z)
        return (score>threshold).astype('int')


class OneVsRest():

    def fit(self, X, t, X_val=None, t_val=None, eta = 0.1, epochs=10, loss_diff=0):
        """X is a Nxm matrix, where N is datapoints and m features, t is target values for X"""

        self.cls = {}
        for c in set(t):
            cl_c = LogRegModel()
            t_c = (t == c).astype('int')
            cl_c.fit(X, t_c, X_val, t_val, eta, epochs, loss_diff)
            self.cls[c] = cl_c

    def predict(self, X):
        cs = []
        scores = []
        for c in self.cls:
            cs.append(c)
            cl = self.cls[c]
            z = add_bias(X)
            score = cl.forward(z)
            scores.append(score)
        p_c = np.argmax(scores, axis=0)
        return np.array([cs[i] for i in p_c])


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

class MLPModel():
    """A multi-layer neural network with multiple hidden layer"""

    def __init__(self,eta = 0.001, dim_hidden = [6]):
        """Initialize the hyperparameters"""
        if type(dim_hidden) == int:
            dim_hidden = [dim_hidden]
        self.eta = eta
        self.dim_hidden = dim_hidden

    def fit(self, X, t, epochs = 100):
        """Initialize the weights. Train *epochs* many epochs."""

        # Initilaization
        y = one_hot_encoding(t)
        dims = [X.shape[1]] + self.dim_hidden + [y.shape[1]]
        self.ws = []

        for i in range(len(dims)-1):
            w_nm = np.random.uniform(-1,1, (dims[i], dims[i+1]))
            self.ws.append(w_nm)

        for e in range(epochs):
            # Forward
            as_ = self.forward(X)
            # Backward
            ws_ = self.backward(y, as_, X)
            self.ws = [w + self.eta*w_ for w,w_ in zip(self.ws, ws_)]

    def forward(self, X):
        """Perform one forward step.
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        X_b = add_bias(X)
        as_ = []
        for w in self.ws:
            w_b = np.r_[np.zeros((1, w.shape[1])), w]
            if as_ == []:
                as_.append(logistic(X_b @ w_b))
            else:
                a_b = add_bias(as_[-1])
                as_.append(logistic(a_b @ w_b))
        return as_

    def backward(self, y, as_, X):
        das = []
        for i in range(len(as_)-1, -1, -1):
            if das == []:
                das.append((y - as_[i])*as_[i]*(1 - as_[i]))
            else:
                das.append(as_[i]*(1 - as_[i])*(das[-1] @ self.ws[i+1].T))
        das = das[::-1]

        ws_ = []
        for i in range(len(das)):
            if i == 0:
                ws_.append(X.T @ das[i])
            else:
                ws_.append(as_[i-1].T @ das[i])

        return ws_

    def predict(self, X_test):
        p = self.forward(X_test)[-1]
        return p.argmax(axis=1)


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

# Data processing
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

def show(X, t=None):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:, 0], X[:,1], c=t, s=20)
    plt.show()

def plot_decision_regions(ax, X, t, clf=[]):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap = 'Paired')

    ax.scatter(X[:,0], X[:,1], c=t, s=20.0, cmap='Paired')

    ax.set_xlim([xx.min(), xx.max()])
    ax.set_ylim([yy.min(), yy.max()])
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")


def __main__():
    datapoints = 2000
    
    # Centers
    cs = np.array([[0,1], [4,1], [8,1], [2,0], [6,0]])
    X, t = get_data(datapoints, cs)

    X_full, t_full = split_data(X,t)
    X_train, X_val, X_test = X_full
    X2_train, X2_val, X2_test = [normalize_feature(X_i) for X_i in X_full]
    t_train, t_val, t_test = t_full

    # Binary classes
    t2_train, t2_val, t2_test = [(t_subset >= 3).astype('int') for t_subset in t_full]

    fig, axs = plt.subplots(2,2)
    plt.tight_layout()

    eta, epochs = 0.075, 100

    # Linear Regression model
    cl_lin_reg = LinRegModel()
    cl_lin_reg.fit(X_train, t2_train, eta=eta, epochs=epochs)
    plot_decision_regions(axs[0,0], X_train, t2_train, cl_lin_reg)
    axs[0,0].set_title("Lin. Reg.")

    # Logistic Regression model
    cl_log_reg = LogRegModel()
    cl_log_reg.fit(X_train, t2_train, eta=eta, epochs=epochs)
    plot_decision_regions(axs[0,1], X_train, t2_train, cl_log_reg)
    axs[0,1].set_title("Log. Reg.")

    # One-vs-Rest
    cl_ovr = OneVsRest()
    cl_ovr.fit(X_train, t_train, eta=eta, epochs=epochs**2)
    plot_decision_regions(axs[1,0], X_train, t_train, cl_ovr)
    axs[1,0].set_title("One-vs-Rest")

    # Multi Layered Perceptron
    cl_mlp = MLPModel(eta=0.01, dim_hidden=[10,5,5])
    cl_mlp.fit(X2_train, t2_train, epochs=3000)
    plot_decision_regions(axs[1,1], X2_train, t2_train, cl_mlp)
    axs[1,1].set_title("MLP")

    plt.show()

if __name__ == "__main__":
    __main__()
