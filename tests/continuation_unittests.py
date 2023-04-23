import unittest
import numpy as np

import sys
import os

# Add SciComp to path
current_file_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_directory, ".."))

from SciComp.bvp import BVP
from SciComp.continuation import bvp_continuation

class TestBVPContinuation(unittest.TestCase):

    def setUp(self):
        # Create a sample BVP object
        a = 0
        b = 1
        N = 10
        alpha = 0
        beta = 0
        D = 1.0
        C = 0
        def q_base(x, u, C=C):
            return C * np.exp(u)
        q_fun = lambda x, u: q_base(x, u, C)

        self.bvp = BVP(a, b, N, alpha, beta, condition_type='Dirichlet', q_fun=q_fun, D=D)

    def test_bvp_continuation_success(self):
        Cs, all_u = bvp_continuation(self.bvp, 0, 4, 200, q_base=lambda x, u, C=0: C * np.exp(u))
        self.assertEqual(len(Cs), 200)
        self.assertEqual(all_u.shape, (200, self.bvp.N + 1))

    def test_bvp_continuation_invalid_method(self):
        with self.assertRaises(ValueError):
            bvp_continuation(self.bvp, 0, 4, 200, q_base=lambda x, u, C=0: C * np.exp(u), method='invalid_method')

    def test_bvp_continuation_invalid_N_C(self):
        with self.assertRaises(ValueError):
            bvp_continuation(self.bvp, 0, 4, -10, q_base=lambda x, u, C=0: C * np.exp(u))

    def test_bvp_continuation_invalid_C_start_C_end(self):
        with self.assertRaises(ValueError):
            bvp_continuation(self.bvp, 4, 0, 200, q_base=lambda x, u, C=0: C * np.exp(u))


if __name__ == '__main__':
    unittest.main()
