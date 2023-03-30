import unittest
import numpy as np

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
        A_DD, b_DD, x_array = bvp.construct_matrix()
        u_array = 0
        u_DD = bvp.solve_matrices(A_DD, b_DD, x_array, u_array)
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
        


if __name__ == '__main__':
    unittest.main()
