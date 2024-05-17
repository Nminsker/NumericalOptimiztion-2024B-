"""
Tests for test_unconstrained_min.py
"""

import unittest
import numpy as np
from src.unconstrained_min import GradientDescent, NewtonMethod
import tests.examples as exmp
from src.utils import contour_plot, line_plot

class TestUnconstrainedMin(unittest.TestCase):
    def runExample(
            self,
            title,
            f,
            x0 = np.array([1.0, 1.0]),
            obj_tol = 1e-12,
            param_tol = 1e-8,
            max_iter = 100,
            x_lim = (-2, 2),
            y_lim = (-2, 2),
            contour_levels = 100,
            file_name=None,
    ):
        gd = GradientDescent()
        nm = NewtonMethod()
        gd_x, gd_fx, gd_min_found = gd.minimize(f, x0.copy(), obj_tol, param_tol, max_iter)
        nm_x, nm_fx, nm_min_found = nm.minimize(f, x0.copy(), obj_tol, param_tol, max_iter)

        print(f"""
--------- Final Results ---------
Gradient Descent: Iterations- {len(gd.path["x"])} | Final Location- {gd_x} |  Final Objective Value- {gd_fx} | Minimum- {gd_min_found}
Newton's Method: Iterations-  {len(nm.path["x"])} | Final Location- {nm_x} |  Final Objective Value- {nm_fx} | Minimum- {nm_min_found}
        """)

        x_paths = {
            "Gradient Descent": tuple(zip(*gd.path["x"])),
            "Newton's Method": tuple(zip(*nm.path["x"]))
        }

        contour_plot(f, x_lim, y_lim, title, x_paths, contour_levels, file_name=file_name)

        fx_paths = {
            "Gradient Descent": gd.path["fx"],
            "Newton's Method": nm.path["fx"],
        }

        line_plot(fx_paths, title)

    def test_f_circle(self):
        self.runExample("Quadratic I", exmp.f_circle,)

    def test_f_ellipse(self):
        self.runExample("Quadratic II", exmp.f_aligned_ellipse,)

    def test_f_ellipse_rotated(self):
        self.runExample("Quadratic III", exmp.f_rotated_ellipse,)

    def test_f_rosenbrock(self):
        self.runExample("Rosenbrock Function", exmp.f_rosenbrock, x0=np.array([-1.0, 2.0]), max_iter=10000, y_lim=(-2, 5))

    def test_f_line(self):
        self.runExample("Linear Function", exmp.f_line, x_lim=(-300, 2), y_lim=(-300, 2))

    def test_f_smooth_triangle(self):
        self.runExample("Exponnent Function", exmp.f_smooth_triangle, x_lim=(-1, 1), y_lim=(-1, 1), contour_levels=50)

if __name__ == '__main__':
    unittest.main()
