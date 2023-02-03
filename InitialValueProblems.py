import numpy as np
import matplotlib.pyplot as plt

from Helpers.solvers import solve_ode

# Define ODE
# dy/dt = y
fun = lambda t, y: y
t0 = 0
y0 = 1
t_max = 20

# Solve ODE
t, y = solve_ode(fun, t0, y0, t_max=t_max, method='Euler', deltat_max=0.1)

# Plot results
# plt.plot(t, y)
# plt.show()

# Define ODE
# d^2y/dt^2 = -y
def fun(t, y):
    u = y[0]
    v = y[1]
    return np.array([v, -u])

t0 = 0
y0 = np.array([1, 0])
t_max = 10

# Solve ODE
t_euler, y_euler = solve_ode(fun, t0, y0, t_max=t_max, method='Euler', deltat_max=0.1)
t_rk4, y_rk4 = solve_ode(fun, t0, y0, t_max=t_max, method='RK4', deltat_max=0.1)
t_heun, y_heun = solve_ode(fun, t0, y0, t_max=t_max, method='Heun', deltat_max=0.1)

# Plot results: y vs y'
plt.plot(y_euler[:, 0], y_euler[:, 1], label='Euler')
plt.plot(y_rk4[:, 0], y_rk4[:, 1], label='RK4')
plt.plot(y_heun[:, 0], y_heun[:, 1], '--', label='Heun')
plt.xlabel('y')
plt.ylabel('dy/dt')
plt.legend()
plt.show()


