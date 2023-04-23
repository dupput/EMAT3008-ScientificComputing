import unittest   # The test framework
import numpy as np

import sys
import os

# Add SciComp to path
current_file_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_directory, ".."))

from SciComp.plotting import plot_phase_plane_2D


class Test_plot_phase_plane_2D(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_plot_phase_plane_2D, self).__init__(*args, **kwargs)

    # Check that plot_phase_plane_2D works
    def test1(self):
        t = np.linspace(0, 10, 100)
        y = np.array([np.sin(t), np.cos(t)])
        
        plot_phase_plane_2D(t, y)
          
            
    # Check transpose of y works
    def test2(self):
        t = np.linspace(0, 10, 100)
        y = np.array([np.sin(t), np.cos(t)])
        
        plot_phase_plane_2D(t, y.T)


    # Test incorrect shape of y
    def test_3(self):
        t = np.linspace(0, 10, 100)
        y = np.array([np.sin(t), np.cos(t)])
        
        with self.assertRaises(ValueError):
            plot_phase_plane_2D(t, y[:-1])

if __name__ == '__main__':
    unittest.main()