import unittest
import numpy as np
try:
    from Helpers.bvp import BVP, InputError
except:
    from bvp import BVP, InputError
class TestBVP(unittest.TestCase):
    def test_boundary_conditions_Dirichlet(self):
        a = 0
        b = 1
        alpha = 0
        N = 100
        
        with self.assertRaises(InputError):
            BVP(a, b, N, alpha, condition_type='Dirichlet')

    def test_boundary_conditions_Neumann(self):
        a = 0
        b = 1
        alpha = 0
        N = 100
        
        with self.assertRaises(InputError):
            BVP(a, b, N, alpha, condition_type='Neumann')

    def test_boundary_conditions_Robin(self):
        a = 0
        b = 1
        alpha = 0
        N = 100
        
        with self.assertRaises(InputError):
            BVP(a, b, N, alpha, condition_type='Robin')

    def test_inputs_a(self):
        a = 'a'
        b = 1
        alpha = 0
        beta = 0
        N = 100
        
        with self.assertRaises(InputError):
            BVP(a, b, N, alpha, beta, condition_type='Dirichlet')

    def test_inputs_b(self):
        a = 0
        b = 'b'
        alpha = 0
        beta = 0
        N = 100
        
        with self.assertRaises(InputError):
            BVP(a, b, N, alpha, beta, condition_type='Dirichlet')

    def test_inputs_N(self):
        a = 0
        b = 1
        alpha = 0
        beta = 0
        N = 3.5
        
        with self.assertRaises(InputError):
            BVP(a, b, N, alpha, beta, condition_type='Dirichlet')
    

    def test_inputs_N2(self):
        a = 0
        b = 1
        alpha = 0
        beta = 0
        N = -2
        
        with self.assertRaises(InputError):
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

        t, dt, C = bvp.time_discretization(t_boundary, t_final, dt)
        solution = bvp.explicit_euler(t)

        idx = np.where(bvp.x_values == 0.5)[0][0]
        numeric = solution[idx, -1]
        
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

        t, dt, C = bvp.time_discretization(t_boundary, t_final, dt)
        solution = bvp.implicit_euler(t)

        idx = np.where(bvp.x_values == 0.5)[0][0]
        numeric = solution[idx, -1]
        
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

        t, dt, C = bvp.time_discretization(t_boundary, t_final, dt)
        solution = bvp.crank_nicolson(t)

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
        methods = ['Scipy Solver', 'Explicit Euler', 'Implicit Euler', 'Crank-Nicolson']
        for method in methods:
            u, t, dt, C = self.bvp.solve_PDE(0, 2, C=0.4, method=method)
            self.assertEqual(u.shape, (self.bvp.N+1, len(t)))
            self.assertTrue(np.all(t >= 0))
            self.assertTrue(np.all(t <= 2))
            self.assertTrue(dt > 0)
            self.assertTrue(C > 0)

    def test_time_step(self):
        u1, t1, dt1, C1 = self.bvp.solve_PDE(0, 2, dt=0.1, C=None)
        u2, t2, dt2, C2 = self.bvp.solve_PDE(0, 2, dt=None, C=0.5)
        self.assertNotEqual(dt1, dt2)

    def test_courant_number(self):
        u1, t1, dt1, C1 = self.bvp.solve_PDE(0, 2, dt=None, C=0.5)
        u2, t2, dt2, C2 = self.bvp.solve_PDE(0, 2, dt=None, C=1)
        self.assertNotEqual(C1, C2)
        


if __name__ == '__main__':
    unittest.main()
