import numpy as np
from help_funcs import one_hot_encoding, add_bias, logistic


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
