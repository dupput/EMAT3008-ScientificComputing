import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from math import ceil
import numpy as np
import time

from SciComp.ivp import solve_ode

def plot_phase_plane_2D(t, y, xsixe=8, ysize=3):
    """
    Plot the phase plane for a 2D ODE.

    Parameters
    ----------
    t : array
        Time array.
    y : array
        Solution array. Must have shape (len(t), 2).

    Returns
    -------
    None

    Examples
    --------
    >>> t = np.linspace(0, 10, 100)
    >>> y = np.array([np.sin(t), np.cos(t)])
    >>> plot_phase_plane_2D(t, y)

    """
    if y.shape[0] == 2:
        y = y.T
    elif y.shape[1] != 2:
        raise ValueError('y must have shape (len(t), 2) or (2, len(t)).')

    if len(t) != len(y):
        raise ValueError('t and y must have the same length.')

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(t, y[:,0], label='x')
    ax[0].plot(t, y[:,1], label='y')
    ax[0].legend()
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('y1, y2')

    ax[1].plot(y[:,0], y[:,1])
    ax[1].set_xlabel('y1')
    ax[1].set_ylabel('y2')
    ax[1].set_aspect('equal', 'box')

    plt.gcf().set_size_inches(xsixe, ysize)

    plt.show()

def plot_continuation(par, x, xsixe=8, ysize=3):
    plt.plot(par, x, '.-', markersize=2)
    plt.xlabel('Parameter')
    plt.ylabel('x')
    # Size
    plt.gcf().set_size_inches(xsixe, ysize)
    
    plt.show()


def animate_PDE(bvp, u, t, xsixe=8, ysize=3):
    """
    Function to animate PDE solution. Only works in Jupyter Notebook.

    Parameters
    ----------
    bvp : BVP
        Boundary value problem object.
    u : array
        Solution array. Must have shape (len(x), len(t)).
    t : array
        Time array.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Animation object.

    Examples
    --------
    >>> f_fun = lambda x, t: np.zeros(len(x))
    >>> q_fun = lambda x, u: 2*u*(1-u)
    >>> bvp = BVP(0, 10, 100, alpha=1, delta=0, condition_type='Neumann', f_fun=f_fun, q_fun=q_fun, D=0.1)
    >>> t, dt, C = bvp.time_discretization(0, 20, C=0.4)
    >>> u = bvp.scipy_solver(t)
    >>> %matplotlib notebook            # This line is needed to render the animation in Jupyter Notebook
    >>> animate_PDE(bvp, u, t)

    """

    # Animate solution using matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    print('Animating...')
    fig = plt.figure()
    # Size
    plt.gcf().set_size_inches(xsixe, ysize)
    ax = plt.axes(xlim=(bvp.a, bvp.b), ylim=(-u.max(), u.max()))
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = bvp.x_values
        y = u[:, i]
        line.set_data(x, y)
        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=len(t), interval=20, blit=True)

    # plt.show()
    return anim


def plot_PDE_fixed_time(x, u, t, analytic_sol):
    """
    plot PDE solution at fixed times

    Parameters
    ----------
    x : array
        x values.
    u : array
        Solution array. Must have shape (len(x), len(t)).
    t : array
        Time array.

    Returns
    -------
    None

    """

    idx = [i*ceil(len(t)/5) for i in range(5)]
    t_checks = t[idx]

    fig = go.Figure()
    for t_check in t_checks:
        idx = np.where(t == t_check)[0][0]
        fig.add_trace(go.Scatter(x=x, y=u[:, idx], name=f't = {t_check}'))
        fig.add_trace(go.Scatter(x=x, y=analytic_sol(x, t_check), name=f'Analytic t = {t_check}', mode='lines', line=dict(dash='dash')))

    fig.show()


def run_comparison_diagnostics(ode, ode_analytical, t0, y0, tf, deltas, methods):
    """
    Compare the accuracy and computation time of different methods for solving
    an ODE. Displays a plot of the error and computation time for each method
    for each step size as well as printing some statistics.

    Parameters
    ----------
    ode : function
        The ODE function.
    fun_analytical : function
        The analytical solution to the ODE.
    t0 : float
        The initial time.
    y0 : array
        The initial conditions.
    tf : float
        The final time.
    deltas : array
        The step sizes to use.
    methods : array
        The methods to use.

    Returns
    -------
    None.
    
    """
    # Create subplots in matplotlib
    # fig = make_subplots(rows=1, cols=2, subplot_titles=('Error', 'Computation time'))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    errors = {
        'Euler': [],
        'RK4': [],
        'Heun': []
    }

    computation_times = {
        'Euler': [],
        'RK4': [],
        'Heun': []
    }

    colors = {
        'Euler': 'blue',
        'RK4': 'red',
        'Heun': 'green'
    }

    for method in methods:
        for delta in deltas:
            start = time.time()
            t_array, y_array = solve_ode(ode, t0, y0, tf, deltat_max=delta, method=method)
            end = time.time()
            error = np.abs(y_array[-1] - ode_analytical(tf))
            computation_time = end - start
            errors[method].append(error[0])
            computation_times[method].append(computation_time)

        # Plot lines and markers
        ax[0].loglog(deltas, errors[method],'-o', label=method, color=colors[method])
        ax[0].set_xlabel('Step size')
        ax[0].set_ylabel('Error')
        ax[0].legend()

        ax[1].loglog(deltas, computation_times[method] ,'-o', label=method, color=colors[method])
        ax[1].set_xlabel('Step size')
        ax[1].set_ylabel('Computation time')
        ax[1].legend()
    
    plt.show()

    # Find the timestep that gives the same error between the Euler, RK4 and Heun methods
    baseline = 0.1
    rk4_dt_mag = int(np.log10(baseline))
    t_, y_ = solve_ode(ode, t0, y0, tf=tf, method='RK4', deltat_max=baseline)
    error = np.abs(y_[-1] - ode_analytical(tf))
    error_mag = int(np.log10(error))

    euler_mag = 10
    while euler_mag >= error_mag:
        baseline /= 10
        t_, y_ = solve_ode(ode, t0, y0, tf=tf, method='Euler', deltat_max=baseline)
        euler_error = np.abs(y_[-1] - ode_analytical(tf))
        euler_mag = int(np.log10(euler_error))
        if euler_mag == error_mag:
            euler_dt = baseline
            break

    euler_dt_mag = int(np.log10(euler_dt))

    baseline = 0.1
    heun_mag = 10
    while heun_mag >= error_mag:
        baseline /= 10
        t_, y_ = solve_ode(ode, t0, y0, tf=tf, method='Heun', deltat_max=baseline)
        heun_error = np.abs(y_[-1] - ode_analytical(tf))
        heun_mag = int(np.log10(heun_error))

    heun_dt_mag = int(np.log10(baseline))
    
    print('_________________________________________________________________________')
    print('Comparison of computation times and errors for a fixed step size ({}):'.format(deltas[3]))
    print('Method:| Computation time (mag):| Error (mag):')
    print('-------|------------------------|-------------')
    for method in methods:
        Error_mag = np.log10(errors[method][3])
        computation_time_mag = np.log10(computation_times[method][3])
        print('{:6} |{:24.2f}|{:12.2f}'.format(method, computation_time_mag, Error_mag))

    print('_________________________________________________________________________')
    print('Comparison of step size required to achieve a fixed magnitude error ({}):'.format(error_mag))
    print('Method:| Step size (mag):')
    print('-------|-----------------')
    print('{:6} |{:17.2f}'.format('Euler', euler_dt_mag))
    print('{:6} |{:17.2f}'.format('RK4', rk4_dt_mag))
    print('{:6} |{:17.2f}'.format('Heun', heun_dt_mag))    

    fig.show()


def visualise_rk4_vs_euler(t_rk4, y_rk4, t_euler, y_euler, y_a):
    """
    Visualise the results of the RK4 and Euler methods. Plots the solution
    against time and the solution against its derivative.

    Parameters
    ----------
    t_rk4 : array
        The time array for the RK4 method.
    y_rk4 : array
        The solution array for the RK4 method.
    t_euler : array
        The time array for the Euler method.
    y_euler : array
        The solution array for the Euler method.
    y_a : array
        The analytical solution.

    Returns
    -------
    None.

    """
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot results: y vs t
    ax1.plot(t_rk4, y_rk4[:, 0], label='RK4', linewidth=3)
    ax1.plot(t_rk4, y_a, '--', label='Analytical', linewidth=3)
    ax1.plot(t_euler, y_euler[:, 0], label='Euler', linewidth=3)
    ax1.set(xlabel='t', ylabel='y')
    ax1.legend()

    # # Plot results: y vs y'
    ax2.plot(y_euler[:, 0], y_euler[:, 1], label='Euler', linewidth=3)
    ax2.plot(y_rk4[:, 0], y_rk4[:, 1], label='RK4', linewidth=3)
    ax2.set(xlabel='y', ylabel='dy/dt')
    ax2.legend()
    plt.show()


def visualise_rk4(t, y, y_a, xsixe=8, ysize=3):
    """
    Visualise the results of the RK4 and Euler methods. Plots the solution
    against time and the solution against its derivative.

    Parameters
    ----------
    t : array
        The time array for the RK4 method.
    y : array
        The solution array for the RK4 method.
    y_a : array
        The analytical solution.

    Returns
    -------
    None.

    """
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot results: y vs t
    ax1.plot(t, y[:, 0], label='RK4', linewidth=3)
    ax1.plot(t, y_a, '--', label='Analytical', linewidth=3)
    ax1.set(xlabel='t', ylabel='y')
    ax1.legend()

    # # Plot results: y vs y'
    ax2.plot(y[:, 0], y[:, 1], label='RK4', linewidth=3)
    ax2.set(xlabel='y', ylabel='dy/dt')
    ax2.legend()

    plt.gcf().set_size_inches(xsixe, ysize)

    plt.show()



def plot_time_series(timeSeries, b_args):
    """
    Plot the time series for each b value. Plots x(t) and y(t) on the left and y(t) vs x(t) on the right.

    Parameters
    ----------
    timeSeries : list
        A list of tuples containing the time and solution arrays for each b value.
    b_args : list
        A list of b values.

    Returns
    -------
    None.
    """
    # Plot results: subplot for each b. Left: x(t), y(t), width: 0.75. Right: dy/dx, width: 0.25
    fig, axs = plt.subplots(len(b_args), 2, gridspec_kw={'width_ratios': [5, 1]}, figsize=(12, 8))

    for i, b in enumerate(b_args):
        # Plot x(t), y(t) on the left. Plot dy/dx on the right.
        axs[i, 0].plot(timeSeries[i][0], timeSeries[i][1][:, 0], label='b = {} \nx(t)'.format(b))
        axs[i, 0].plot(timeSeries[i][0], timeSeries[i][1][:, 1], label='y(t)')
        axs[i, 0].set_xlabel('t')
        axs[i, 0].set_ylabel('b = {}'.format(b))

        axs[i, 1].plot(timeSeries[i][1][:, 0], timeSeries[i][1][:, 1], label='b = {} \ndy/dx'.format(b))
        axs[i, 1].set_aspect('equal')
        axs[i, 1].set_xlabel('x')
        axs[i, 1].set_ylabel('y')
    # Plot size
    plt.show()


def plot_bvp(bvp, u_linear, u_analytical, xsixe=8, ysize=3):
    """
    Plot the solution of the BVP and the analytical solution.

    Parameters
    ----------
    bvp : BVP
        The BVP object.
    u_linear : array
        The solution of the BVP using the linear method.
    u_analytical : array
        The analytical solution.

    Returns
    -------
    None.
    """
    plt.plot(bvp.x_values, u_linear, label='Numerical solution')
    plt.plot(bvp.x_values, u_analytical, label='Analytic solution', linestyle='--')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.gcf().set_size_inches(xsixe, ysize)

    plt.show()


def plot_max_u_vs_C(Cs, all_u, xsixe=8, ysize=3):
    """
    Plot the maximum value of u(x) vs C.

    Parameters
    ----------
    all_u : array
        The solution array for all C values.
    Cs : array
        The array of C values.

    Returns
    -------
    None.
    """
    max_u = np.max(all_u, axis=1)

    plt.plot(Cs[max_u != 0], max_u[max_u != 0])
    plt.xlabel('C')
    plt.ylabel('Maximum value of u(x)')
    plt.gcf().set_size_inches(xsixe, ysize)

    plt.show()


def plot_PDE(bvp, u, t):
    """
    Plot the solution of the PDE.

    Parameters
    ----------
    bvp : BVP
        The BVP object.
    u : array
        The solution of the PDE.
    t : array
        The time array.

    Returns
    -------
    None.
    """ 

    # Two subplots, one axes is 3d and the other is 2d
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # Create a meshgrid of 2D arrays for t and bvp.x_values
    T, BVP_X = np.meshgrid(t, bvp.x_values)

    # Plot the surface.
    surf = ax.plot_surface(BVP_X, T, u, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)      

    # Customize the z axis.
    ax.set_zlim(-0.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')

    # Now plot the 2D plot at t = 5
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(bvp.x_values, u[:, 100])
    ax.set_xlabel(f'x (t = {t[100]:.2f})')
    ax.set_ylabel('u')
    plt.tight_layout()

    plt.show()