from lin_reg import *
from log_reg import *
from one_vs_rest import *
from mlp import *
from help_funcs import *
from data_processing import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_decision_regions(ax, X, t, clf=[]):
    """Plots decision regions on subplot ax."""
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
