from math import ceil 
import numpy as np
import warnings

import os
import sys

# Add SciComp to path
current_file_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_directory, ".."))


from SciComp._bvp_base_class import base_BVP
from SciComp._pde_solvers import *
from SciComp._ode_solvers import *


class FunctionError(Exception):
    """Exception raised for errors in the function."""
    pass

class BVP(base_BVP):
    """
    Class for solving boundary value problems. Inherits from base_BVP class.

    All constructor and helper functions are inherited from base_BVP class. The only front facing
    functions are the two handlers, solve_ode and solve_pde, which call the appropriate functions to 
    solve the problem.

    Parameters
    ----------
    a : float
        Left boundary of the domain: u(a) = alpha.
    b : float
        Right boundary of the domain: u(b) = beta.
    N : int
        Number of grid points. Must be an integer greater than 0.
    alpha : float
        Value of u(a), left boundary condition.
    beta : float, optional
        Value of u(b), right boundary condition for Dirichlet boundary conditions. Must be specified
        for Dirichlet boundary conditions. The default value is None.
    delta : float, optional
        Value of u'(b), right boundary condition for Neumann boundary conditions. Must be specified
        for Neumann boundary conditions. The default value is None.
    gamma : float, optional
        Value of u''(b), right boundary condition for Robin boundary conditions. Must be specified
        for Robin boundary conditions. The default value is None.
    condition_type : str, optional
        Type of boundary conditions. Must be either 'Dirichlet', 'Neumann' or 'Robin'. The default
        value is 'Dirichlet'.
    q_fun : function, optional
        Source function for the differential equation. Must be of the form q_fun(x, u). Default value
        is None.
    f_fun : function, optional
        Function to define the boundary conditions. Default value is None.
    D_const : float, optional
        Constant coefficient in the differential equation. Default value is None.
    sparse : bool, optional
        Boolean indicating whether to use sparse matrices or not. Default value is False.
    """
    def __init__(self, a, b, N, alpha, beta=None, delta=None, gamma=None, condition_type=None, 
                 q_fun=None, f_fun=None, D=1, sparse=False):
        super().__init__(a, b, N, alpha, beta, delta, gamma, condition_type, q_fun, f_fun, D, sparse)


    def solve_ODE(self, u_array=None, method='linear'):
        """
        Function handler to solve the boundary value problem for a fixed time value. BVP is solved using either the
        Scipy's root solver for nonlinear problems or numpy's linalg solver for linear problems. If the q_fun contains 
        a nonlinear term, then the Scipy's root solver should be selected. If the q_fun is linear, then the numpy's 
        linalg solver should be selected.

        Parameters
        ----------
        u_array : numpy.ndarray
            Array of u values corresponding to the initial guess for the solution. Must be consistent with with the boundary conditions. Must be 
            provided if method is 'nonlinear'. The default is None.
        method : str, optional
            Method to solve the boundary value problem. If method is 'linear', then the numpy's linalg solver is used. If method is 'nonlinear',
            then the Scipy's root solver is used. The default is 'linear'.

        Returns
        -------
        solution : numpy.ndarray
            Solution to the boundary value problem. solution is a vector of size x_array.
        success : bool
            Boolean indicating whether the solver was successful or not.
        """
        # Sparse matrix only works with linear method
        if self.sparse and method != 'linear':
            raise ValueError('Sparse matrices can only be used with linear method.')

        # Method check
        METHODS = {'linear': solve_matrices, 'tridiagonal': solve_thomas_algorithm,
                   'nonlinear': solve_nonlinear}
        if method not in METHODS:
            raise ValueError('Invalid method: {}. Method must be one of {}.'.format(method, METHODS.keys()))
        else:
            method = METHODS[method]

        # Prepare the matrices
        A, b, x_array = self.construct_matrix()

        # Check arrays are consistent
        if u_array is None:
            u_array = np.zeros(self.shape)
            
        elif len(u_array) != len(x_array) or len(u_array) != len(b) or len(u_array) != len(A):
            message = """
            The length of the initial guess array must be consistent with the length of the x_array, b and A.
            For {} conditions, the length of the initial guess array must be {}, ().""".format(self.condition_type, self.shape, len(x_array))
            raise ValueError(message)

        # Solve the boundary value problem
        u, success = method(self, A, b, u_array=u_array, x_array=x_array)

        if not success:
            warnings.warn('The solver was not successful. The solution may not be accurate.')

        # Concatanate the solution with the boundary conditions
        u = self.concatanate(u, type='ODE')

        return u, success
    
    
    def solve_PDE(self, t_boundary, t_final, dt=None, C=None, method='Crank-Nicolson'):
        """
        Function to solve the PDE.

        Parameters
        ----------
        t_boundary : float
            Initial time.
        t_final : float
            Final time.
        dt : float, optional
            Time step. The default is None.
        C : float, optional
            Courant number. The default is None.
        method : str, optional
            Method to use to solve the PDE. The default is 'Crank-Nicolson'.

        Returns
        -------
        u : numpy.ndarray
            Solution to the PDE.
        t : numpy.ndarray
            Array of time values.
        dt : float
            Time step.
        C : float
            Courant number.

        Examples
        --------
        >>> a = 0
        >>> b = 1
        >>> alpha = 0
        >>> beta = 0
        >>> f_fun = lambda x, t: np.sin(np.pi * (x - a) / (b - a))
        >>> D = 0.1
        >>> N = 100

        >>> bvp = BVP(a, b, N, alpha, beta, D=D, condition_type='Dirichlet', f_fun=f_fun)
        
        >>> t_boundary = 0
        >>> C = 0.5
        >>> t_final = 2
        >>> u = bvp.solve_PDE(t_boundary, t_final, C=C, method='Crank-Nicolson')

        """
        # Type checks
        if not isinstance(t_boundary, (int, float)):
            raise TypeError('t_boundary must be a number')
        if not isinstance(t_final, (int, float)):
            raise TypeError('t_final must be a number')
        if not isinstance(dt, (int, float, type(None))):
            raise TypeError('dt must be a number')

        if dt is None and C is None:
            raise ValueError('Either dt or C must be provided')
        elif dt is not None and C is not None:
            raise ValueError('Only one of dt or C must be provided')

        # Discretize time
        t, dt, C = self.time_discretization(t_boundary, t_final, dt=dt, C=C, method=method)

        # Sparse matrix only supported for Crank-Nicolson, Implicit Euler
        if self.sparse and method not in ['Crank-Nicolson', 'Implicit Euler']:
            raise ValueError('Sparse matrix only supported for Crank-Nicolson and Implicit Euler')

        # Method check. Imports are from _pde_solvers.py
        METHODS = {'Scipy Solver' : scipy_solver,
                   'Explicit Euler': lambda self, t: explicit_method(self, t, method='Euler'),
                   'Explicit RK4': lambda self, t: explicit_method(self, t, method='RK4'),
                   'Explicit Heun': lambda self, t: explicit_method(self, t, method='Heun'),
                   'Implicit Euler': implicit_euler,
                   'Crank-Nicolson': crank_nicolson,
                   'IMEX Euler': imex_euler}
        if method not in METHODS:
            raise ValueError('Invalid method: {}. Method must be one of {}.'.format(method, METHODS.keys()))
        else:
            method = METHODS[method]

        # Solve the PDE
        u = method(self, t)

        u = self.concatanate(u, type='PDE', t=t)
        return u, t, dt, C    
    
    
if __name__ == '__main__':
    # Define PDE 
    a = 0
    b = 1
    alpha = 0
    beta = 0
    def f_fun(X, T): 
        return np.sin(np.pi * (X - a) / (b - a))
    D = 1
    N = 100

    bvp = BVP(a, b, N, alpha, beta, D=D, condition_type='Dirichlet', f_fun=f_fun)

    t_boundary = 0
    C = 0.5
    t_final = 2

    t, dt, C = bvp.time_discretization(t_boundary, t_final, C=C, method='Explicit Euler')
    A_, b_, x_array = bvp.construct_matrix()


    # u = explicit_method(bvp, t, method='Euler')
    u, t, dt, C = bvp.solve_PDE(t_boundary, t_final, C=C, method='Explicit RK4')
    
    # Get analytical solution
    analytic_sol = lambda x, t: np.exp(-D * np.pi**2 * t / (b-a)**2) * np.sin(np.pi * (x - a) / (b - a))

    for index in [1000, 3000, 5000, 7000]:
        numerical_sol = u[:, index]
        analytical_sol = analytic_sol(bvp.x_values, t[index])

        import matplotlib.pyplot as plt
        plt.plot(bvp.x_values, numerical_sol, label='Numerical')
        plt.plot(bvp.x_values, analytical_sol, '--', label='Analytic')

    plt.show()