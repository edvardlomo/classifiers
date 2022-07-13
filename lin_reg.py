import numpy as np
from help_funcs import add_bias

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
