import numpy as np
import math


class InteriorPointMinimization:

    def __init__(self, obj_tol=1e-12, param_tol=1e-8, max_inner_iter=100, max_outer_iter=20, epsilon=1e-10):
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_inner_iter = max_inner_iter
        self.max_outer_iter = max_outer_iter
        self.epsilon = epsilon
        
    #define unchangable properties
    @property
    def WOLFE_COND_CONST(self):
        return 0.01
    
    @property
    def BACKTRACKING_CONST(self):
        return 0.5

    @property
    def T(self):
        return 1
    
    @property
    def MU(self):
        return 10

    def wolfe(self, func, direction, x):
        """ Wolfe condition for backtracking line search
            :param func: function to be minimized
            :param direction: search direction
            :param x: current point
            :return: step length
        """
        alpha = 1.0
        fx = func(x, False)[0]
        dot_grad = np.dot(func(x, False)[1], direction)

        while alpha > 1e-6 and (func(x + alpha * direction, False)[0] > fx + self.WOLFE_COND_CONST * alpha * dot_grad):
            alpha *= self.BACKTRACKING_CONST

        return alpha

    def phi(self, constraints, x):
        """ phi barrier function for inequality constraints
            :param constraints: list of inequality constraints
            :param x: current point
            :return: phi value, gradient, hessian
            """
        
        f, g, h = 0, 0, 0
        for constraint in constraints:
            f_x, g_x, h_x = constraint(x, True)

            f += math.log(-f_x)

            grad = g_x / f_x
            g += grad

            dim = grad.shape[0]

            grad_mesh = np.tile(grad.reshape(dim, -1), (1, dim)) * np.tile(grad.reshape(dim, -1).T, (dim, 1))
            h += (h_x * f_x - grad_mesh) / f_x ** 2

        return -f, -g, -h

    def construct_block_matrix(self,hessian, eq_const_mat):
        A_mat = hessian
        if eq_const_mat.size:
            dim = eq_const_mat.shape[0]
            A_mat = np.block(
            [
                [hessian, eq_const_mat.T],
                [eq_const_mat, np.zeros((dim, dim))]
            ]
        )
            
        return A_mat

    def _apply_phi_vals(self, phi_val, phi_grad, phi_hess, t, obj_val, obj_grad, obj_hess):
        def addVal(base, phi):
            return t * base + phi
        
        return addVal(obj_val, phi_val), addVal(obj_grad, phi_grad), addVal(obj_hess, phi_hess)
    
    def _update_history(self, x, objective, history_dict):
        history_dict["x"].append(x)
        history_dict["objective"].append(objective)

    def get_direction(self, hessian, eq_constraint_matrix, gradient):
        """ Returns the serach direction
            :param hessian: hessian matrix
            :param eq_constraint_matrix: equality constraints matrix
            :param gradient: gradient
        """
        block_matrix = self.construct_block_matrix(hessian, eq_constraint_matrix)
        eq_vector = np.concatenate([-gradient, np.zeros(block_matrix.shape[0] - len(gradient))])
        search_direction = np.linalg.solve(block_matrix, eq_vector)[: len(gradient)]
        return search_direction
    
    def interior_pt(
        self,
        objective_func,
        x0,
        inequality_constraints,
        equality_constraints_matrix,
    ):
        curr_x = x0
        t = self.T
        self.history = {"x": [], "objective": []}
        self.outer_history = {"x": [], "objective": []}

        objective_value, gradient, hessian = objective_func(curr_x, True)
        self._update_history(curr_x, objective_value, self.history)
        self._update_history(curr_x, objective_value, self.outer_history)

        phi_value, phi_gradient, phi_hessian = self.phi(inequality_constraints, curr_x)
        objective_value, gradient, hessian = self._apply_phi_vals(phi_value, phi_gradient, phi_hessian, t, objective_value, gradient, hessian)


        for _ in range(self.max_outer_iter):
            search_direction = self.get_direction(hessian, equality_constraints_matrix, gradient)

            previous_x, previous_objective_value = float("inf"), float("inf")

            for _ in range(self.max_inner_iter):
                if np.sum(np.abs(curr_x - previous_x)) < self.param_tol:
                    break

                lambda_value = np.sqrt(search_direction @ (hessian @ search_direction.T))

                if 0.5 * lambda_value ** 2 < self.obj_tol or previous_objective_value - objective_value < self.obj_tol:
                    break

                alpha = self.wolfe(objective_func, search_direction, curr_x )

                previous_x, previous_objective_value = curr_x, objective_value

                curr_x = curr_x + alpha * search_direction
                objective_value, gradient, hessian = objective_func(curr_x, True)
                phi_value, phi_gradient, phi_hessian = self.phi(inequality_constraints, curr_x)

                self._update_history(curr_x, objective_value, self.history)

                objective_value, gradient, hessian = self._apply_phi_vals(phi_value, phi_gradient, phi_hessian, t, objective_value, gradient, hessian)

            self._update_history(curr_x, (objective_value - phi_value) / t, self.outer_history)

            if len(inequality_constraints) / t < self.epsilon:
                break

            t *= self.MU
            previous_x, previous_objective_value = curr_x, objective_value
