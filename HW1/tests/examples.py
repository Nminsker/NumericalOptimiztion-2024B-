"""
This module contains objective functions to be used for testing the optimization algorithms
"""
import numpy as np

def _calc_quadratic(Q, x, eval=False):
    """ Helper function to calculate the value, gradient and hessian of a quadratic function
    params:
    ------
        Q: array-like, the quadratic coefficient matrix
        x: array-like, the input vector
        eval: bool, whether to evaluate the hessian or not
    """
    f_x = x @ Q @ x
    g_x = 2 * Q @ x
    h_x = 2 * Q if eval else None

    return f_x, g_x, h_x

def f_circle(x, eval=False):
    """ Returns f(x), gradient of f(x) and hessian of f(x) for a quadratic function with a circular level set"""
    Q = np.array([[1, 0], [0, 1]])
    return _calc_quadratic(Q, x, eval)

def f_aligned_ellipse(x, eval=False):
    """ Returns f(x), gradient of f(x) and hessian of f(x) for a quadratic function with an aligned ellipse level set"""
    Q = np.array([[1, 0], [0, 100]])
    return _calc_quadratic(Q, x, eval)

def f_rotated_ellipse(x, eval=False):
    """ Returns f(x), gradient of f(x) and hessian of f(x) for a quadratic function with a rotated ellipse level set"""
    Q = np.array([[np.sqrt(3)/2, 0.5], [-0.5, np.sqrt(3)/2]]) @ np.array([[100,0],[0,1]]) @ np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    return _calc_quadratic(Q, x, eval)

def f_rosenbrock(x, eval=False):
    """ Returns f(x), gradient of f(x) and hessian of f(x) for the Rosenbrock function"""
    x1, x2 = tuple(x)
    f_x = 100*(x2 - (x1**2))**2 + (1-x1)**2
    g_x = np.array([

        -400*x1*x2 + 400*(x1**3) + 2*x1 - 2,
        200*x2 - 200*(x1**2)
    ])
    h_x = np.array([
        [-400*x2 + 1200*(x1**2) + 2 , -400*x1], 
        [-400*x1                  , 200]
    ]) if eval else None
    
    return f_x, g_x, h_x

def f_line(x, eval=False):
    """ Returns f(x), gradient of f(x) and hessian of f(x) for a linear function"""
    a = np.array([3, 3])
    f_x = a @ x
    g_x = a
    h_x = np.zeros((2, 2)) if eval else None

    return f_x, g_x, h_x

def f_smooth_triangle(x, eval=False):
    """ Returns f(x), gradient of f(x) and hessian of f(x) for a function with a smooth triangle level set"""
    x1, x2 = tuple(x)
    f_x = np.exp(x1+3*x2-0.1) + np.exp(x1-3*x2-0.1) + np.exp(-x1-0.1)
    g_x = np.array([
        np.exp(x1+3*x2-0.1) + np.exp(x1-3*x2-0.1) - np.exp(-x1-0.1),
        3*np.exp(x1+3*x2-0.1) - 3*np.exp(x1-3*x2-0.1)
    ])
    h_x =  np.array([
            [
                np.exp(x1+3*x2-0.1) + np.exp(x1-3*x2-0.1) + np.exp(-x1-0.1),
                3*np.exp(x1+3*x2-0.1) - 3*np.exp(x1-3*x2-0.1)
            ],
            [
                3*np.exp(x1+3*x2-0.1) - 3*np.exp(x1-3*x2-0.1),
                9*np.exp(x1+3*x2-0.1) + 9*np.exp(x1-3*x2-0.1)
            ]
        ]) if eval else None

    return f_x, g_x, h_x