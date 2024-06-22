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

def plot_iterations(
        title,
        obj_values_1=None,
        obj_values_2=None,
):
    """
    Plots the objective function values over iterations for two methods.
    :param title:
    :param obj_values_1:
    :param obj_values_2:
    :param label_1:
    :param label_2:
    :return:
    """
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Function Value")

    if obj_values_1 is not None:
        valid_values_1 = [v for v in obj_values_1 if v is not None]
        ax.plot(range(len(valid_values_1)), valid_values_1, label="Inner Objective Value")

    if obj_values_2 is not None:
        valid_values_2 = [v for v in obj_values_2 if v is not None]
        ax.plot(range(len(valid_values_2)), valid_values_2, label="Outer Objective Value")

    ax.legend()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def plot_feasible_set_2d(path_points=None):
    """
    Plots the feasible region for a given 2D problem and the path points
    of an algorithm if provided.

    Parameters:
    - path_points: List of tuples [(x1, y1), (x2, y2), ...] representing
                   the path points of an algorithm. Default is None.
    """
    d = np.linspace(-2, 4, 300)
    x, y = np.meshgrid(d, d)

    plt.imshow(
        ((y >= -x + 1) & (y <= 1) & (x <= 2) & (y >= 0)).astype(int),
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        cmap="Greys",
        alpha=0.3,
    )

    x_line = np.linspace(0, 4, 2000)
    y1 = -x_line + 1
    y2 = np.ones(x_line.size)
    y3 = np.zeros(x_line.size)
    x_boundary = np.ones(x_line.size) * 2

    plt.plot(x_line, y1, 'b-', label=r"$y = -x + 1$")
    plt.plot(x_line, y2, 'g-', label=r"$y = 1$")
    plt.plot(x_line, y3, 'r-', label=r"$y = 0$")
    plt.plot(x_boundary, x_line, 'm-', label=r"$x = 2$")

    x_path, y_path = zip(*path_points)
    plt.plot(x_path, y_path, label="Algorithm's Path", color="k", marker=".", linestyle="--")
    plt.xlim(0, 3)
    plt.ylim(0, 2)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title("Feasible Region")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feasible_set_3d(path_points=None):
    d = np.linspace(0, 1, 100)
    x, y = np.meshgrid(d, d)
    z = 1 - x - y
    z = np.where(z >= 0, z, np.nan)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, alpha=0.5, color='blue')

    if path_points is not None and path_points.shape[1] == 3:
        ax.plot(
            path_points[:, 0],
            path_points[:, 1],
            path_points[:, 2],
            'o-',
            color='black',
        )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if path_points is not None:
        final_point = path_points[-1]
        convergence_value = np.sum(final_point)
        title = f'Value at point of convergence: {convergence_value:.3}, constraint holds'
    else:
        title = 'Feasible Region in 3D'

    ax.set_title(title)
    plt.show()