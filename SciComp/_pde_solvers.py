import numpy as np
from scipy.integrate import solve_ivp


def scipy_solver(bvp, t):
    """
    Function to solve the PDE using Scipy's solve_ivp.

    Parameters
    ----------
    bvp : BVP object
        Boundary value problem to solve. Object instantiated in SciComp/bvp.py.
    t : numpy.ndarray
        Array of time values.

    Returns
    -------
    u : numpy.ndarray
        Solution to the PDE.
    """
    A, b, x_array = bvp.construct_matrix()
    
    u_boundary = bvp.f_fun(x_array, t[0])

    if bvp.q_fun is None:
        def PDE(t, u, D, A, b):
            return D / bvp.dx ** 2 * (A @ u + b)
    else:
        def PDE(t, u, D, A, b):
            return D / bvp.dx ** 2 * (A @ u + b) + bvp.q_fun(x_array, u)  

    sol = solve_ivp(PDE, (t[0], t[-1]), u_boundary, method='RK45', t_eval=t, args=(bvp.D, A, b))

    u = sol.y

    return u


def explicit_euler(bvp, t):
    """
    Function to solve the PDE using the explicit Euler method.

    Parameters
    ----------
    bvp : BVP object
        Boundary value problem to solve. Object instantiated in SciComp/bvp.py.
    t : numpy.ndarray
        Array of time values.

    Returns
    -------
    solution : numpy.ndarray
        Solution to the PDE.
    """
    A, b, x_array = bvp.construct_matrix()

    u = np.zeros((len(x_array), len(t)))
    u[:, 0] = bvp.f_fun(x_array, t[0])
    u[0, :] = bvp.alpha
    u[-1, :] = bvp.beta

    # Loop over time
    for ti in range(0, len(t)-1):
        for xi in range(1, len(x_array)-1):
            if xi == 0:
                # u[0, ti+1] = u[0, ti] + bvp.C * (bvp.alpha - 2 * u[0, ti] + u[1, ti])
                u[xi, ti+1] = bvp.alpha
            elif xi > 0 and xi < bvp.N:
                u[xi, ti+1] = u[xi, ti] + bvp.C * (u[xi-1, ti] - 2 * u[xi, ti] + u[xi+1, ti])
            else:
                u[xi, ti+1] = u[xi, ti] + bvp.C * (bvp.beta - 2 * u[xi, ti] + u[xi-1, ti])

    return u


def implicit_euler(bvp, t):
    """
    Function to solve the PDE using the implicit Euler method.

    Parameters
    ----------
    bvp : BVP object
        Boundary value problem to solve. Object instantiated in SciComp/bvp.py.
    t : numpy.ndarray
        Array of time values.

    Returns
    -------
    solution : numpy.ndarray
        Solution to the PDE.
    """
    A, b, x_array = bvp.construct_matrix()

    I = np.eye(bvp.shape)

    lhs = I - bvp.C * A
    rhs = bvp.C * b

    u = np.zeros((len(x_array), len(t)))
    u[:, 0] = bvp.f_fun(x_array, t[0])

    for ti in range(0, len(t)-1):
        u[:, ti+1] = np.linalg.solve(lhs, u[:, ti] + rhs)

    return u


def crank_nicolson(bvp, t):
    """
    Function to solve the PDE using the Crank-Nicolson method.

    Parameters
    ----------
    bvp : BVP object
        Boundary value problem to solve. Object instantiated in SciComp/bvp.py.
    t : numpy.ndarray
        Array of time values.

    Returns
    -------
    solution : numpy.ndarray
        Solution to the PDE.
    """
    A, b, x_array = bvp.construct_matrix()

    I = np.eye(bvp.shape)

    lhs = I - bvp.C * A / 2
    rhs = I + bvp.C * A / 2

    u = np.zeros((len(x_array), len(t)))
    u[:, 0] = bvp.f_fun(x_array, t[0])

    for ti in range(0, len(t)-1):
        u[:, ti+1] = np.linalg.solve(lhs, rhs @ u[:, ti] + bvp.C * b)

    return u