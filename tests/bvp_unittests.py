import unittest
import numpy as np

import sys
import os

# Add SciComp to path
current_file_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_directory, ".."))

from SciComp.bvp import BVP

class TestBVP(unittest.TestCase):
    def test_boundary_conditions_Dirichlet(self):
        a = 0
        b = 1
        alpha = 0
        N = 100
        
        with self.assertRaises(ValueError):
            BVP(a, b, N, alpha, condition_type='Dirichlet')

    def test_boundary_conditions_Neumann(self):
        a = 0
        b = 1
        alpha = 0
        N = 100
        
        with self.assertRaises(ValueError):
            BVP(a, b, N, alpha, condition_type='Neumann')

    def test_boundary_conditions_Robin(self):
        a = 0
        b = 1
        alpha = 0
        N = 100
        
        with self.assertRaises(ValueError):
            BVP(a, b, N, alpha, condition_type='Robin')

    def test_inputs_a(self):
        a = 'a'
        b = 1
        alpha = 0
        beta = 0
        N = 100
        
        with self.assertRaises(ValueError):
            BVP(a, b, N, alpha, beta, condition_type='Dirichlet')

    def test_inputs_b(self):
        a = 0
        b = 'b'
        alpha = 0
        beta = 0
        N = 100
        
        with self.assertRaises(ValueError):
            BVP(a, b, N, alpha, beta, condition_type='Dirichlet')

    def test_inputs_N(self):
        a = 0
        b = 1
        alpha = 0
        beta = 0
        N = 3.5
        
        with self.assertRaises(ValueError):
            BVP(a, b, N, alpha, beta, condition_type='Dirichlet')
    

    def test_inputs_N2(self):
        a = 0
        b = 1
        alpha = 0
        beta = 0
        N = -2
        
        with self.assertRaises(ValueError):
            BVP(a, b, N, alpha, beta, condition_type='Dirichlet')

    
    def test_example_1(self):
        a = 0
        b = 5
        n = 10
        alpha = 0
        beta = 50
        condition_type = 'Dirichlet'
        q_fun = lambda x, u: 9.8

        bvp = BVP(a, b, n, alpha, beta, condition_type=condition_type, q_fun=q_fun)
        
        u_DD, success = bvp.solve_ODE()
        
        y_n1 = -9.8*bvp.dx**2 + 2*u_DD[0] - u_DD[1]
        
        velocity = (u_DD[1] - y_n1) / (2*bvp.dx)

        self.assertAlmostEqual(velocity, 34.5)


    def test_example_2(self):
        exact = np.exp(-0.2 * np.pi**2)
        a = 0
        b = 1
        alpha = 0
        beta = 0
        N = 100
        D = 0.1

        f_fun = lambda x, t: np.sin(np.pi * x)
        bvp = BVP(a, b, N, alpha, beta, D=D, condition_type='Dirichlet', f_fun=f_fun)

        t_boundary = 0
        dt = 0.0001
        t_final = 2

        u, t, dt, C = bvp.solve_PDE(t_boundary, t_final, dt=dt, method='Explicit Euler')

        idx = np.where(bvp.x_values == 0.5)[0][0]
        numeric = u[idx, -1]
        
        assert np.isclose(numeric, exact, atol=1e-2)

    def test_example_3(self):
        exact = np.exp(-0.2 * np.pi**2)
        a = 0
        b = 1
        alpha = 0
        beta = 0
        N = 100
        D = 0.1

        f_fun = lambda x, t: np.sin(np.pi * x)
        bvp = BVP(a, b, N, alpha, beta, D=D, condition_type='Dirichlet', f_fun=f_fun)

        t_boundary = 0
        dt = 0.0001
        t_final = 2

        u, t, dt, C = bvp.solve_PDE(t_boundary, t_final, dt=dt, method='Implicit Euler')

        idx = np.where(bvp.x_values == 0.5)[0][0]
        numeric = u[idx, -1]
        
        assert np.isclose(numeric, exact, atol=1e-2)

    def test_example_4(self):
        exact = np.exp(-0.2 * np.pi**2)
        a = 0
        b = 1
        alpha = 0
        beta = 0
        N = 100
        D = 0.1

        f_fun = lambda x, t: np.sin(np.pi * x)
        bvp = BVP(a, b, N, alpha, beta, D=D, condition_type='Dirichlet', f_fun=f_fun)

        t_boundary = 0
        dt = 0.0001
        t_final = 2

        solution, t, dt, C = bvp.solve_PDE(t_boundary, t_final, dt=dt, method='Crank-Nicolson')

        idx = np.where(bvp.x_values == 0.5)[0][0]
        numeric = solution[idx, -1]
        
        assert np.isclose(numeric, exact, atol=1e-2)

    def test_solve_ODE_1(self):
        a = 0
        b = 1
        N = 10
        alpha = 0
        beta = 0
        condition_type = 'Dirichlet'
        q_fun = lambda x, u: 1
        bvp = BVP(a, b, N, alpha, beta, condition_type=condition_type, q_fun=q_fun)
        u_tridiag, success = bvp.solve_ODE(method='tridiagonal')
        u_linear, success = bvp.solve_ODE(method='linear')
        u_nonlinear, success = bvp.solve_ODE(method='nonlinear')
        def u_analytic(x):
            return -1/2 * (x - a) * (x - b) + ((beta - alpha) / (b - a)) * (x - a) + alpha
        x_values = bvp.x_values
        u_analytic = u_analytic(x_values)
        # Assert that the solutions are the same
        assert np.allclose(u_tridiag, u_linear, u_nonlinear, u_analytic)

    def test_solve_ODE_2(self):
        a = 0
        b = 1
        N = 10
        alpha = 0
        beta = 10
        condition_type = 'Dirichlet'
        q_fun = lambda x, u: 0

        bvp = BVP(a, b, N, alpha, beta, condition_type=condition_type, q_fun=q_fun)
        u, success = bvp.solve_ODE()

        def u_analytic(x):
            return (beta - alpha) / (b - a) * (x - a) + alpha

        x_values = bvp.x_values
        u_analytic = u_analytic(x_values)

        # Assert that the solutions are the same
        assert np.allclose(u, u_analytic)


    def setUp(self):
        # Initialize a BVP instance with the given parameters
        a = 0
        b = 1
        alpha = 0
        beta = 0
        f_fun = lambda x, t: np.sin(np.pi * (x - a) / (b - a))
        D = 0.1
        N = 100
        
        self.bvp = BVP(a, b, N, alpha, beta, D=D, condition_type='Dirichlet', f_fun=f_fun)
        self.bvp_sparse = BVP(a, b, N, alpha, beta, D=D, condition_type='Dirichlet', f_fun=f_fun, sparse=True)

    def test_type_errors(self):
        with self.assertRaises(TypeError):
            self.bvp.solve_PDE('0', 2)
        with self.assertRaises(TypeError):
            self.bvp.solve_PDE(0, '2')
        with self.assertRaises(TypeError):
            self.bvp.solve_PDE(0, 2, dt='0.1')


    def test_method_value_error(self):  
        with self.assertRaises(ValueError):
            self.bvp.solve_PDE(0, 2, method='InvalidMethod')


    def test_methods(self):
        methods = ['Scipy Solver', 'Explicit Euler', 'Explicit RK4', 'Explicit Heun', 'Implicit Euler', 'Crank-Nicolson']
        for method in methods:
            u, t, dt, C = self.bvp.solve_PDE(0, 2, C=0.4, method=method)
            self.assertEqual(u.shape, (self.bvp.N+1, len(t)))
            self.assertTrue(np.all(t >= 0))
            self.assertTrue(np.all(t <= 2))
            self.assertTrue(dt > 0)
            self.assertTrue(C > 0)

    def test_sparse(self):
        methods = ['Crank-Nicolson', 'Implicit Euler']
        for method in methods:
            u, t, dt, C = self.bvp.solve_PDE(0, 2, C=0.4, method=method)
            u_sparse, t_sparse, dt_sparse, C_sparse = self.bvp_sparse.solve_PDE(0, 2, C=0.4, method=method)
            self.assertTrue(np.allclose(u, u_sparse))


    def test_time_step(self):
        u1, t1, dt1, C1 = self.bvp.solve_PDE(0, 2, dt=0.1, C=None)
        u2, t2, dt2, C2 = self.bvp.solve_PDE(0, 2, dt=None, C=0.5)
        self.assertNotEqual(dt1, dt2)

    def test_courant_number(self):
        u1, t1, dt1, C1 = self.bvp.solve_PDE(0, 2, dt=None, C=0.5)
        u2, t2, dt2, C2 = self.bvp.solve_PDE(0, 2, dt=None, C=1)
        self.assertNotEqual(C1, C2)

    def test_constructor(self):
        instance = BVP(a=0, b=1, N=10, alpha=1, beta=0, condition_type='Dirichlet')
        self.assertEqual(instance.a, 0)
        self.assertEqual(instance.b, 1)
        self.assertEqual(instance.N, 10)
        self.assertEqual(instance.alpha, 1)

    def test_invalid_interval(self):
        with self.assertRaises(ValueError):
            BVP(a=1, b=0, N=10, alpha=1)

    def test_invalid_number_of_points(self):
        with self.assertRaises(ValueError):
            BVP(a=0, b=1, N=-1, alpha=1)
 


if __name__ == '__main__':
    unittest.main()
