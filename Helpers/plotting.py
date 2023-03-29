import matplotlib.pyplot as plt
from matplotlib import animation
import plotly.graph_objects as go
from math import ceil
import numpy as np

def plot_phase_plane_2D(t, y):
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

    plt.show()


def animate_PDE(bvp, u, t):
    print('Animating...')
    fig = plt.figure()
    ax = plt.axes(xlim=(bvp.a, bvp.b), ylim=(-1.5, 1.5))
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