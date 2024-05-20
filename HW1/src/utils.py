"""
This module contains utility functions.
"""
import matplotlib.pyplot as plt
import numpy as np

MARK_BY_ALGO = {
    "Gradient Descent": {"color": "r", "marker": "*"},
    "Newton's Method": {"linestyle": "dashed", "color": "b", "marker": "o"},
}

def contour_plot(f, x_lim, y_lim, title, paths, levels=100, file_name=None):
    x = np.linspace(x_lim[0], x_lim[1])
    y = np.linspace(y_lim[0], y_lim[1])

    xs, ys = np.meshgrid(x, y)
    f_vals = np.vectorize(lambda x1, x2: f(
        np.array([x1, x2]), False)[0])(xs, ys)

    fig, ax = plt.subplots(1, 1)
    contour = ax.contourf(x, y, f_vals, levels)
    fig.colorbar(contour)
    ax.set_title(f"{title}: Algorithm Paths")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    if paths:
        for algoName, (x_vals, y_vals) in paths.items():
            plt.plot(x_vals, y_vals, **MARK_BY_ALGO[algoName] , label=algoName,)
        plt.legend()

    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()
        print()

def line_plot(values_dict, title):
    fig, ax = plt.subplots(1, 1)
    for algoName, values in values_dict.items():
        x = np.linspace(1, len(values), len(values))
        ax.plot(x, values, **MARK_BY_ALGO[algoName], label=algoName)
    ax.set_title(f"{title}: Function values vs iteration number")
    ax.set_xlabel('# Iteration')
    ax.set_ylabel('f(x)')
    plt.legend()
    plt.show()
