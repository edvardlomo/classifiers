import numpy as np
from log_reg import *


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
