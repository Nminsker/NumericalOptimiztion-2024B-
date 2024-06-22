import unittest
import numpy as np
from src.constrained_min import InteriorPointMinimization
from src.utils import plot_iterations, plot_feasible_set_2d, plot_feasible_set_3d
from tests.examples import *


class TestInteriorPointMethod(unittest.TestCase):

    def _display_convergence_info(self, x, fx, problem_description, constraints, print_sum=True):
        
        print(f"{problem_description}, Point of convergence: ({','.join([f'x{i}' for i,_ in enumerate(x)])})={x}")
        print(f"{problem_description}, Value at point of convergence: {fx}")


        for i, constraint in enumerate(constraints, start=1):
            constraint_value = constraint(x, False)[0]
            print(f"Inequality constraint {i} Value at point of convergence: {constraint_value}")

        if print_sum:
            print(f"Sum of variables at point of convergence: {sum(x)}")

    def run_test(
            self,
            problem_description,
            func,
            start_point,
            constraints,
            eq_constraint_mat,
            plot_feas_func,
    ):
        minimizer = InteriorPointMinimization()
        minimizer.interior_pt(func, start_point, constraints, eq_constraint_mat)

        self._display_convergence_info(minimizer.history["x"][-1], minimizer.history["objective"][-1], problem_description, constraints)

        plot_iterations(
            f"Convergence of {problem_description}",
            obj_values_1=minimizer.history["objective"],
            obj_values_2=minimizer.outer_history["objective"],
        )

        plot_feas_func(np.array(minimizer.history["x"]))

    def test_qp(self):
        eq_constraint_mat = np.array([1, 1, 1]).reshape(1, -1)
        constraints = [qp_ineq_constraint_1, qp_ineq_constraint_2, qp_ineq_constraint_3]
        self.run_test(
            "Quadratic Problem",
            qp,
            np.array([0.1, 0.2, 0.7], dtype=np.float64),
            constraints,
            eq_constraint_mat,
            plot_feas_func=plot_feasible_set_3d
        )

    def test_lp(self):
        constraints = [
            lp_ineq_constraint_1,
            lp_ineq_constraint_2,
            lp_ineq_constraint_3,
            lp_ineq_constraint_4,
        ]

        self.run_test(
            "Linear Problem", 
            lp,
            np.array([0.5, 0.75], dtype=np.float64),
            constraints, 
            np.array([]),
            plot_feasible_set_2d,    
        )


if __name__ == "__main__":
    unittest.main()