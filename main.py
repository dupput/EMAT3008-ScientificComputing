import numpy as np
import matplotlib.pyplot as plt

from solvers import solve_ode


# Define ODE parameters
fun = lambda t, y: 2 - np.exp(-4*t) - 2*y
t0 = 0
y0 = 1
t_max = 3

# Solve ODE
t, y = solve_ode(fun, t0, y0, t_max=t_max, method='Euler', deltat_max=0.1)


