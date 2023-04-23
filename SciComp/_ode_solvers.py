import numpy as np
from scipy.optimize import root

def euler_step(fun, t, y, h):
    """Perform one step of the Euler method.
    
    parameters
    ----------
    fun : function
        Function f(t, y) to integrate.
    t : float
        Current value of t.
    y : float
        Current value of y.
    h : float
        Step size.
        
    returns
    -------
    t : float
        New value of t.
    y : float
        New value of y.
    """
    m = fun(t, y)
    t = t + h
    y = y + h*m

    return t, y


def rk4_step(fun, t, y, h):
    """Perform one step of the Runge-Kutta method of order 4.
    
    parameters
    ----------
    fun : function
        Function f(t, y) to integrate.
    t : float
        Current value of t.
    y : float
        Current value of y.
    h : float
        Step size.
        
    returns
    -------
    t : float
        New value of t.
    y : float
        New value of y.
    """
    k1 = fun(t, y)
    k2 = fun(t + h/2, y + h*k1/2)
    k3 = fun(t + h/2, y + h*k2/2)
    k4 = fun(t + h, y + h*k3)
    t = t + h
    y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)

    return t, y


def heun_step(fun, t, y, h):
    """Perform one step of the Heun method.
    
    parameters
    ----------
    fun : function
        Function f(t, y) to integrate.
    t : float
        Current value of t.
    y : float
        Current value of y.
    h : float
        Step size.
        
    returns
    -------
    t : float
        New value of t.
    y : float
        New value of y.
    """
    k1 = fun(t, y)
    k2 = fun(t + h, y + h*k1)
    t = t + h
    y = y + h/2*(k1 + k2)

    return t, y


def solve_matrices(bvp, A, b, u_array=None, x_array=None):
    """
    Function to solve the linear system of equations Au = -b - q(x, u) * dx^2.
    The function q(x, u) is defined by the user and must be linear in u.

    Parameters
    ----------
    bvp : BVP object
        Boundary value problem to solve. Object instantiated in SciComp/bvp.py.
    A : numpy.ndarray
        Matrix A. Must be size consistent with b, x_array and u_array.
    b : numpy.ndarray
        Vector b. Must be consistent with A, x_array and u_array.
    u_array : numpy.ndarray, optional
        Array of u values corresponding to A and b. Must be consistent with A, b and x_array. 
        Must be provided if q_fun is provided. The default is None.
    x_array : numpy.ndarray, optional
        Array of x values corresponding to A and b. Must be consistent with A, b and u_array. 
        Must be provided if q_fun is provided. The default is None.

    Returns
    -------
    solution : numpy.ndarray
        Solution to the linear system of equations. solution is a vector of size x_array.
    success : bool
        Boolean indicating whether the linear system of equations was successfully solved.
    """
    
    # Add the source term to the vector b
    if bvp.q_fun is not None:
        if x_array is None or u_array is None:
            raise ValueError('x_array and u_array must be provided if q_fun is provided')
        else:
            b = b + bvp.q_fun(x_array, u_array) * bvp.dx**2

    try:        
        # Solve the linear system
        u = np.linalg.solve(A, -b)
        success = True

        return u, success
    
    except:
        success = False
        return None, success
    

def solve_thomas_algorithm(bvp, A, b, u_array=None, x_array=None):
    """
    Function to solve the linear system of equations Au = -b - q(x, u) * dx^2.
    The function q(x, u) is defined by the user and must be linear in u.
    The function uses the Thomas algorithm to solve the linear system of equations:

        a_i * u_{i-1} + b_i * u_i + c_i * u_{i+1} = d_i

    where
        d_i = -b_i - q(x_i, u_i) * dx^2

    Parameters
    ----------
    bvp : BVP object
        Boundary value problem to solve. Object instantiated in SciComp/bvp.py.
    A : numpy.ndarray
        Matrix A. Must be size consistent with b, x_array and u_array.
    b : numpy.ndarray
        Vector b. Must be consistent with A, x_array and u_array.
    u_array : numpy.ndarray, optional
        Array of u values corresponding to A and b. Must be consistent with A, b and x_array. 
        Must be provided if q_fun is provided. The default is None.
    x_array : numpy.ndarray, optional
        Array of x values corresponding to A and b. Must be consistent with A, b and u_array. 
        Must be provided if q_fun is provided. The default is None.

    Returns
    -------
    solution : numpy.ndarray
        Solution to the linear system of equations. solution is a vector of size x_array.
    success : bool
        Boolean indicating whether the linear system of equations was successfully solved.
    """
    
    # Add the source term to the vector b and define the vector d
    if bvp.q_fun is not None:
        if x_array is None or u_array is None:
            raise ValueError('x_array and u_array must be provided if q_fun is provided')
        else:
            d = -(b + bvp.q_fun(x_array, u_array) * bvp.dx**2)
    else:
        d = -b

    try:        
        # define the vectors a, b and c
        a = A.diagonal(-1).copy()
        b = A.diagonal().copy()
        c = A.diagonal(1).copy()

        # Forward sweep
        for i in range(1, bvp.shape):
            m = a[i-1]/b[i-1]
            b[i] = b[i] - m * c[i-1]
            d[i] = d[i] - m * d[i-1]

        # Backward sweep
        u = np.zeros(bvp.shape)
        u[-1] = d[-1]/b[-1]
        for i in range(bvp.shape-2, -1, -1):
            u[i] = (d[i] - c[i] * u[i+1])/b[i]

        success = True
        return u, success
    
    except Exception as e:
        success = False
        return u_array, success


def solve_nonlinear(bvp, A, b, u_array, x_array=None):
    """
    Function to solve the nonlinear system of equations Au = -b - q(x, u) * dx^2.
    The function q(x, u) is defined by the user and can be nonlinear in u.

    Parameters
    ----------
    bvp : BVP object
        Boundary value problem to solve. Object instantiated in SciComp/bvp.py.
    A : numpy.ndarray
        Matrix A. Must be size consistent with b, x_array and u_array.
    b : numpy.ndarray
        Vector b. Must be consistent with A, x_array and u_array.
    u_array : numpy.ndarray
        Array of u values corresponding to A and b. Must be consistent with A, b and x_array.
    x_array : numpy.ndarray, optional
        Array of x values corresponding to A and b. Must be consistent with A, b and u_array.
        Must be provided if q_fun is provided. The default is None.

    Returns
    -------
    solution : numpy.ndarray
        Solution to the nonlinear system of equations. solution is a vector of size x_array.
    success : bool
        Boolean indicating whether the solver was successful or not.
    """
    
    if bvp.q_fun is None:
        if x_array is None:
            raise ValueError('x_array and u_array must be provided if q_fun is not provided')
        else:
            func = lambda u_array: bvp.D * A @ u_array + b
    else:
        func = lambda u_array: bvp.D * A @ u_array + b + bvp.q_fun(x_array, u_array) * bvp.dx**2
    
    sol = root(func, u_array)
    u = sol.x
    
    return u, sol.success