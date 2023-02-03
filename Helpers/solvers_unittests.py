import unittest   # The test framework
import numpy as np

from solvers import solve_to

class Test_TestSplitInput(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_TestSplitInput, self).__init__(*args, **kwargs)
        self.fun = lambda t, y: 2 - np.exp(-4*t) - 2*y
        self.t0 = 0
        self.y0 = 1

    # Testing the function solve_ode
    def test_1(self):
        with self.assertRaises(ValueError):
            solve_to(self.fun, self.t0, self.y0, method='euler')

    
if __name__ == '__main__':
    unittest.main()
