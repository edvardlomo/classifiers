import numpy as np
from help_funcs import add_bias, logistic


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
