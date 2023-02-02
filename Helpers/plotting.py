import numpy as np
import matplotlib.pyplot as plt

from solvers import solve_ode

def plot_variations(fun, t0, y0, t_max):
    # Experiment with different step sizes and methods
    deltas = [0.1, 0.01]
    t = []
    y = []

    for method in ['Euler', 'RK4']:
        for delta in deltas:
            t_, y_ = solve_ode(fun, t0, y0, t_max=t_max, method=method, deltat_max=delta)
            t.append(t_)
            y.append(y_)
            plt.plot(t_, y_, label='{}, delta={}'.format(method, delta))


    anayltical = lambda t: 1 + 1/2*np.exp(-4*t) - 1/2*np.exp(-2*t)
    t_a = np.linspace(t0, t_max, 100)
    y_a = anayltical(t_a)
    plt.plot(t_a, y_a, label='Analytical')

    # Plot labels
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.show()



    