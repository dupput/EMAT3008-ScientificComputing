import unittest   # The test framework
import numpy as np

from solvers import solve_to, shooting, InputError, FunctionError

class Test_solve_to(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_solve_to, self).__init__(*args, **kwargs)

    # Testing the function solve_ode
    def test_1(self):
        fun = lambda t, y: y
        t0 = 0
        y0 = 1
        t_max = 1
        with self.assertRaises(ValueError):
            solve_to(fun, t0, y0, method='euler')

    # Checking that the function solve_ode returns the correct value
    def test_2(self):
        fun = lambda t, y: y
        analytic = lambda t: np.exp(t)
        t0 = 0
        y0 = 1
        t_max = 1
        t, y = solve_to(fun, t0, y0, t_max=t_max, method='RK4')
        self.assertTrue(np.allclose(y[-1], analytic(t_max)))

# Testing the function shooting
class Test_shooting(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_shooting, self).__init__(*args, **kwargs)

    # Testing format of initial_guess
    def test_1(self):
        def ode(t, y, a=1, d=0.1, b=0.1):
            x = y[0]
            y = y[1]
            dxdt = x*(1-x) - (a*x*y)/(d+x)
            dydt = b*y*(1 - y/x)
            return np.array([dxdt, dydt])

        def phase_function(t, y):
            dx, dy = ode(t, y)
            return dx

        initial_guess = np.array([0.6, 0.8, 0.9, 35])

        with self.assertRaises(InputError):
            shooting(initial_guess, ode, phase_function)

    # Checking that the function shooting returns the correct value
    def test_2(self):
        def ode(t, y, a=1, d=0.1, b=0.1):
            x = y[0]
            y = y[1]
            dxdt = x*(1-x) - (a*x*y)/(d+x)
            dydt = b*y*(1 - y/x)
            return np.array([dxdt, dydt])

        def phase_function(t, x, y, a=1, d=0.1, b=0.1):
            dx = x*(1-x) - (a*x*y)/(d+x)
            return dx

        initial_guess = np.array([0.6, 0.8, 35])
        
        with self.assertRaises(FunctionError):
            shooting(initial_guess, ode, phase_function)

    # Checking that the function shooting returns roughly correct values
    def test_3(self):
        def ode(t, y, a=1, d=0.1, b=0.1):
            x = y[0]
            y = y[1]
            dxdt = x*(1-x) - (a*x*y)/(d+x)
            dydt = b*y*(1 - y/x)
            return np.array([dxdt, dydt])

        def phase_function(t, y):
            dx, dy = ode(t, y)
            return dx

        initial_guess = np.array([0.6, 0.8, 35])
        X0, T = shooting(initial_guess, ode, phase_function)
        t, y = solve_to(ode, 0, X0, t_max=T, method='RK4')

        # Check that the phase function is close to zero at the end of the
        # integration and that the final state is close to the initial state
        self.assertTrue(np.isclose(phase_function(T, X0), 0, atol=1e-3))
        self.assertTrue(np.allclose(X0, y[-1], atol=1e-3))

    # Test to raise error if solution is not found
    def test_4(self):
        def ode(t, y, a=1, d=0.1, b=0.1):
            x = y[0]
            y = y[1]
            dxdt = x*(1-x) - (a*x*y)/(d+x)
            dydt = b*y*(1 - y/x)
            return np.array([dxdt, dydt])

        def phase_function(t, y):
            dx, dy = ode(t, y)
            return dx

        initial_guess = np.array([20, 0.4, 35])
        with self.assertRaises(ValueError):
            shooting(initial_guess, ode, phase_function)


if __name__ == '__main__':
    unittest.main()
