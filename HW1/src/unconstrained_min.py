"""
This module contains the implementation of the unconstrained optimization
"""
import numpy as np

class LineSearchBase:
    CALC_HESSIAN = False
    NAME = "LineSearchBase"

    """ Base class for the line search optimization algorithms """
    def __init__(
            self,
            c1 = 0.01,
            backtracking = 0.5,
        ):
        self.C1 = c1
        self.BACKTRACKING = backtracking
              
    def wolfe_condition(self, f, x, fx, gx, direction):
        """
        Implement the Wolfe condition to ensure that the step length satisfies the Armijo rule.
        Returns the step length that satisfies the 1st Wolfe condition .

        params:
        ------
            f: callable, the function to be minimized
            x: array-like, the current location
            fx: float, the current value of the function at x, i.e. f(x)
            gx: array-like, the gradient of the function at x, i.e. grad(x)
            px: array-like, the direction to move in, i.e. px
        """
        alpha = 1.0
        gx_px_c1 = self.C1 * gx @ direction

        while f(x + alpha * direction)[0] > fx + alpha *gx_px_c1:
            alpha *= self.BACKTRACKING
            if alpha < 1e-62:
                break

        return alpha

    def _print_iteration(self, i, x, f_x, total_iter):
        """ Helper function to print iteration information """
        print(f"{i:>4}/{total_iter} | x: {x} | f(x): {f_x}")

    def calc_px(self, gx=None, hx=None):
        """ calcularte px (step direction) for the linear solve
        override to implment """
        raise NotImplementedError()
    
    def updatePath(self, x, fx):
        self.path["x"].append(x)
        self.path["fx"].append(fx)

    def minimize(self, f, x0, obj_tol, param_tol, max_iter):
        """
        Run the a linear solver algorithm to minimize the function f.
        Returns (final location, final objective value, success flag)

        params:
        ------
            f: callable, the function to be minimized
            x0: array-like, the initial guess (i.e. starting point)
            obj_tol: float, the numeric tolerance for successful convergence in terms of small enough change in objective function values, between two consecutive iterations (f(x_{i+1}) and f(x_i)
            param_tol: float, the numeric tolerance for successful termination in terms of small enough distance between two consecutive iterations iteration locations (x_{i+1} and x_i)
            max_iter: int, the maximum number of iterations to run
        """
        self.path = {
            "x": [],
            "fx": []
        }

        min_found = False
        x = x0
        curr_fx, curr_gx, curr_hx = f(x, self.CALC_HESSIAN)

        print(f"----------{self.NAME}----------")
        self._print_iteration(1, x, curr_fx, max_iter)
        self.updatePath(x.copy(), curr_fx)

        for i in range(2, max_iter+1):
            px = self.calc_px(hx=curr_hx, gx=curr_gx)
            if px is None:
                break

            step_len = self.wolfe_condition(f, x, curr_fx, curr_gx, px)
            
            # update x
            prev_x = x.copy()
            x += step_len * px

            # update value and gradient
            prev_fx = curr_fx
            curr_fx, curr_gx, curr_hx = f(x, self.CALC_HESSIAN)

            self.updatePath(x.copy(), curr_fx)
            self._print_iteration(i, x, curr_fx, max_iter)


            # check for convergence
            if abs(curr_fx - prev_fx) < obj_tol or np.linalg.norm(x - prev_x) < param_tol: #or not curr_gx.any():
                min_found = True
                break

        
        return x, curr_fx, min_found


class GradientDescent(LineSearchBase):
    """ Gradient Descent optimization algorithm """
    
    NAME = "Gradient Descent"

    def calc_px(self, gx, hx=None):
        return -gx



class NewtonMethod(LineSearchBase):
    """ Newton's Method optimization algorithm """

    CALC_HESSIAN = True
    NAME = "Newton's Method"

    def calc_px(self, gx, hx):
        try:
                px = -np.linalg.pinv(hx) @ gx
        except np.linalg.LinAlgError:
            # if the hessian is not invertible, we can't proceed
            px = None
        return px
    
    # def minimize(self, f, x0, obj_tol, param_tol, max_iter):
    #     self.path = {
    #         "x": [],
    #         "fx": []
    #     }

    #     x = np.array(x0, dtype=float)
    #     x_history = [x0]
        
    #     epsilon = 1e-6

    #     f_prev = float("inf")
    #     x_prev = x.copy()

    #     for iteration in range(max_iter):
    #         f_x, g_x, h_x = f(x, True)
    #         self.updatePath(x_prev, f_x)

    #         if iteration > 0:
    #             if np.sum(np.abs(x - x_prev)) < param_tol:
    #                 return x, f_x, x_history, True
    #             if (f_prev - f_x) < obj_tol:
    #                 return x, f_x, x_history, True

    #         if self.method == "Newton":
    #             regularized_hessian = h_x + epsilon * np.eye(h_x.shape[0])
    #             h_x_inv = np.linalg.pinv(regularized_hessian)
    #             p = -np.matmul(h_x_inv, g_x)

    #             # newton decrement
    #             lambda_squared = np.dot(p, np.dot(h_x, p))



    
