import matplotlib.pyplot as plt
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

