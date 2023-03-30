import unittest   # The test framework
import numpy as np

from Helpers.solvers import solve_to, shooting, InputError, FunctionError

class Test_solve_to(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_solve_to, self).__init__(*args, **kwargs)


    # Checking that the function solve_ode returns the correct value
    def test_1(self):
        fun = lambda t, y: y
        analytic = lambda t: np.exp(t)
        t0 = 0
        y0 = np.array(1)
        tf = 1
        t, y = solve_to(fun, t0, y0, tf=tf, method='RK4')
        self.assertTrue(np.allclose(y[-1], analytic(tf)))

    # Check function errors are caught
    def test_2(self):
        fun = 'Not a function'
        t0 = 0
        y0 = 1
        tf = 1
        with self.assertRaises(FunctionError):
            solve_to(fun, t0, y0, tf=tf, method='RK4')

    # Check mismatching y0 and fun outputs
    def test_3(self):
        def fun(t, y, beta=1, sigma=-1):
            u1 = y[0]
            u2 = y[1]
            u3 = y[2]
            du1dt = beta * u1 - u2 + sigma * u1 * (u1**2 + u2**2)
            du2dt = u1 + beta * u2 + sigma * u2 * (u1**2 + u2**2)
            du3dt = -u3
            return np.array([du1dt, du2dt])
        t0 = 0
        y0 = np.array([1, 1, 1])
        tf = 1

        with self.assertRaises(FunctionError):
            solve_to(fun, t0, y0, tf=tf, method='RK4')

    # Check mismatching y0 and fun outputs
    def test_4(self):
        def fun(t, y, beta=1, sigma=-1):
            u1 = y[0]
            u2 = y[1]
            u3 = y[2]
            du1dt = beta * u1 - u2 + sigma * u1 * (u1**2 + u2**2)
            du2dt = u1 + beta * u2 + sigma * u2 * (u1**2 + u2**2)
            du3dt = -u3
            return [du1dt, du2dt, du3dt]
        t0 = 0
        y0 = np.array([1, 1, 1])
        tf = 1

        with self.assertRaises(FunctionError):
            solve_to(fun, t0, y0, tf=tf, method='RK4')

    # Check if incorrect input type
    def test_5(self):
        fun = lambda t, y: y
        t0 = 0
        y0 = 1
        tf = 10
        with self.assertRaises(InputError):
            solve_to(fun, t0, y0, tf=tf, method='RK4')
    

    # Check if no tf or n_max is given
    def test_6(self):
        fun = lambda t, y: y
        t0 = 0
        y0 = np.array(1)
        with self.assertRaises(ValueError):
            solve_to(fun, t0, y0, method='RK4')

    # Check that tf > t0
    def test_7(self):
        fun = lambda t, y: y
        t0 = 0
        y0 = np.array(1)
        tf = -5
        with self.assertRaises(ValueError):
            solve_to(fun, t0, y0, tf=tf, method='RK4')

    def test_8(self):
        fun = lambda t, y: y
        t0 = 0
        y0 = np.array(1)
        n_max = -5
        with self.assertRaises(ValueError):
            solve_to(fun, t0, y0, n_max=n_max)

    # Testing the function solve_ode
    def test_9(self):
        fun = lambda t, y: y
        t0 = 0
        y0 = np.array(1)
        tf = 1
        with self.assertRaises(ValueError):
            solve_to(fun, t0, y0, method='euler')
    
    # Test t0 value
    def test_10(self):
        fun = lambda t, y: y
        t0 = 'Not a number'
        y0 = np.array(1)
        tf = 2
        with self.assertRaises(ValueError):
            solve_to(fun, t0, y0, tf=tf)

    # Test n_max value
    def test_11(self):
        fun = lambda t, y: y
        t0 = 0
        y0 = np.array(1)
        n_max = 0.5
        with self.assertRaises(ValueError):
            solve_to(fun, t0, y0, n_max=n_max)

        # Test n_max value
    def test_12(self):
        fun = lambda t, y: y
        t0 = 0
        y0 = np.array(1)
        n_max = 'Not a number'
        with self.assertRaises(ValueError):
            solve_to(fun, t0, y0, n_max=n_max)

        # Test n_max value
    def test_13(self):
        fun = lambda t, y: y
        t0 = 0
        y0 = np.array(1)
        tf = 'Not a number'
        with self.assertRaises(ValueError):
            solve_to(fun, t0, y0, n_max=tf)

    # Test delta_t value
    def test_14(self):
        fun = lambda t, y: y
        t0 = 0
        y0 = np.array(1)
        tf = 1
        deltat_max = -1
        with self.assertRaises(ValueError):
            solve_to(fun, t0, y0, tf=tf, deltat_max=deltat_max)

    def test_15(self):
        fun = lambda t, y: y
        t0 = 0
        y0 = np.array(1)
        tf = 1
        deltat_max = 0
        with self.assertRaises(ValueError):
            solve_to(fun, t0, y0, tf=tf, deltat_max=deltat_max)

    def test_16(self):
        fun = lambda t, y: y
        t0 = 0
        y0 = np.array(1)
        tf = 1
        deltat_max = 'Not a number'
        with self.assertRaises(ValueError):
            solve_to(fun, t0, y0, tf=tf, deltat_max=deltat_max)
# ----------------------------------------------------------------------------- #

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
        t, y = solve_to(ode, 0, X0, tf=T, method='RK4')

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
