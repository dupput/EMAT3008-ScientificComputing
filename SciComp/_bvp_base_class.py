from math import ceil 
import warnings

import numpy as np
import scipy.sparse as sparse


from SciComp._pde_solvers import *
from SciComp._ode_solvers import *

class FunctionError(Exception):
    """Exception raised for errors in the function."""
    pass


class base_BVP():
    """
    Class for solving boundary value problems.

    Functions inside the class have been split into distinct sections for clarity. The sections are:
    - Check functions
    - BVP construction functions
    - ODE solvers
    - PDE solvers

    The only front facing functions are the two handlers, solve_ode and solve_pde, which call the
    appropriate functions to solve the problem.

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
        If True, use sparse matrices in the PDE solver. Default value is False.
    """
    def __init__(self, a, b, N, alpha, beta=None, delta=None, gamma=None, condition_type=None, 
                 q_fun=None, f_fun=None, D=1, sparse=False):
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.N = N
        self.condition_type = condition_type
        self.q_fun = q_fun
        self.f_fun = f_fun
        self.D = D
        self.sparse = sparse

        # Construct the grid
        self.grid_discretisation()

        # Check that the boundary conditions are valid
        self.check_boundary_validity()

        # Check q_fun and f_fun
        self.check_functions()


    # -------------------------------- Check functions --------------------------------
    def check_boundary_validity(self):
        """
        Check that the boundary conditions are valid. Conditions are:
        - alpha and beta must be specified for Dirichlet boundary conditions
        - alpha and delta must be specified for Neumann boundary conditions
        - alpha, delta and gamma must be specified for Robin boundary conditions

        Returns
        -------
        None.
        """

        if self.condition_type == 'Dirichlet':
            if self.alpha is None or self.beta is None:
                raise ValueError('alpha and beta must be specified for Dirichlet boundary conditions')

        elif self.condition_type == 'Neumann':
            if self.alpha is None or self.delta is None:
                raise ValueError('alpha and delta must be specified for Neumann boundary conditions')
            
        elif self.condition_type == 'Robin':
            if self.alpha is None or self.delta is None or self.gamma is None:
                raise ValueError('alpha, delta and gamma must be specified for Robin boundary conditions')
            
        else:
            raise ValueError('condition_type must be either Dirichlet or Neumann or Robin')


    def check_interval(self):
        """
        Check that a and b are numbers and that a < b.
        """
        if not isinstance(self.a, (int, float)) or not isinstance(self.b, (int, float)):
            raise TypeError("Both a and b must be numbers.")
            
        if self.a >= self.b:
            raise ValueError("The value of a must be less than b. a = {}, b = {}".format(self.a, self.b))


    def check_number_of_points(self):
        """
        Check that N is a positive integer.
        """
        if not isinstance(self.N, int) or self.N <= 0:
            raise ValueError("N must be a positive integer. N = {}".format(self.N))


    def check_functions(self):
        """
        Check that q_fun and f_fun are functions.

        Returns
        -------
        None.
        """
        if self.q_fun is not None:
            if not callable(self.q_fun):
                raise ValueError('q_fun must be a function')
        
        if self.f_fun is not None:
            if not callable(self.f_fun):
                raise ValueError('f_fun must be a function')


    def grid_discretisation(self):
        """
        Create the grid. The grid is a numpy array of N+1 points between a and b.

        Returns
        -------
        None.
        """
        # Check if a and b are numbers
        if not isinstance(self.a, (int, float)) or not isinstance(self.b, (int, float)):
            raise ValueError('a and b must be numbers. a = {}, b = {}'.format(self.a, self.b))

        # Check if N is an integer
        if not isinstance(self.N, int) or self.N < 0:
            raise ValueError('N must be an positive integer. N = {}'.format(self.N))

        self.x_values = np.linspace(self.a, self.b, self.N + 1)
        self.dx = (self.b - self.a) / self.N

        return self


    def dirichlet_boundary_conditions(self):
        """
        Function to define the matrix A and the vector b for the Dirichlet boundary conditions.

        Takes the general form of the central difference scheme:
        A:  [-2. 1.  0. ...  0.  0.]  b:   [alpha]
            [1. -2.  1. ...  0.  0.]       [0.]
            [0.  1. -2. ...  0.  0.]       [0.]
            [    ...         ...   ]       [...]
            [0.  0.  0. ... -2.  1.]       [0.]
            [0.  0.  0. ...  1. -2.]       [beta]

        Returns
        -------
        A : numpy.ndarray
            Matrix A. A is a (N-1)x(N-1) matrix.
        b : numpy.ndarray
            Vector b. b is a (N-1) vector.
        x_array : numpy.ndarray
            Array of x values corresponding to A and b. x_array is a (N-1) vector.
        """

        N = self.N
        delta_x = self.dx

        # Initialize the matrix A and the vector b
        A = np.zeros((N - 1, N - 1))
        b = np.zeros(N - 1)

        # Populate the matrix A according to the central difference scheme
        for i in range(1, N-2):
            A[i, i - 1] = 1
            A[i, i] = -2
            A[i, i + 1] = 1

        A[0, 0] = -2
        A[0, 1] = 1
        A[N-2, N-2] = -2
        A[N-2, N - 3] = 1
        b[0] = self.alpha
        b[N-2] = self.beta

        # Create the array of x values that will be used in solvers
        x_array = self.x_values[1:-1]

        return A, b, x_array


    def neumann_boundary_conditions(self):
        """
        Function to define the matrix A and the vector b for the Neumann boundary conditions.

        Takes the general form of the central difference scheme:
        A:  [-2. 1.  0. ...  0.  0.]  b:   [alpha]
            [1. -2.  1. ...  0.  0.]       [0.]
            [0.  1. -2. ...  0.  0.]       [0.]
            [    ...         ...   ]       [...]
            [0.  0.  0. ... -2.  1.]       [0.]
            [0.  0.  0. ...  2. -2.]       [2 * delta * delta_x]

        Returns
        -------
        A : numpy.ndarray
            Matrix A. A is a NxN matrix.
        b : numpy.ndarray
            Vector b. b is a N vector.
        x_array : numpy.ndarray
            Array of x values corresponding to A and b. x_array is a N vector.
        """
        N = self.N
        delta_x = self.dx

        # Initialize the matrix A and the vector b
        A = np.zeros((N, N))
        b = np.zeros(N)

        # Populate the matrix A according to the central difference scheme
        for i in range(1, N-1):
            A[i, i - 1] = 1
            A[i, i] = -2
            A[i, i + 1] = 1

        A[0, 0] = -2
        A[0, 1] = 1
        A[N-1, N-1] = 2
        A[N-1, N-2] = -2
        b[0] = self.alpha
        b[N-1] = 2 * self.delta * delta_x

        # Create the array of x values that will be used in solvers
        x_array = self.x_values[1:]

        return A, b, x_array


    def robin_boundary_conditions(self):
        """
        Function to define the matrix A and the vector b for the Robin boundary conditions.

        Takes the general form of the central difference scheme:
        A:  [-2. 1.  0. ...  0.            0.           ]  b:   [alpha]
            [1. -2.  1. ...  0.            0.           ]       [0.]
            [0.  1. -2. ...  0.            0.           ]       [0.]
            [    ...         ...                        ]       [...]
            [0.  0.  0. ... -2.            1.           ]       [0.]
            [0.  0.  0. ...  2. -2*(1 + gamma * delta_x)]       [2 * gamma * delta_x]
        
        Returns
        -------
        A : numpy.ndarray
            Matrix A. A is a NxN matrix.
        b : numpy.ndarray
            Vector b. b is a N vector.
        x_array : numpy.ndarray
            Array of x values corresponding to A and b. x_array is a N vector.
        """
        N = self.N
        delta_x = self.dx

        # Initialize the matrix A and the vector b
        A = np.zeros((N, N))
        b = np.zeros(N)

        # Populate the matrix A according to the central difference scheme
        for i in range(1, N-1):
            A[i, i - 1] = 1
            A[i, i] = -2
            A[i, i + 1] = 1

        A[0, 0] = -2
        A[0, 1] = 1
        A[N-1, N-1] = -2*(1 + self.gamma * delta_x)
        A[N-1, N-2] = 2
        b[0] = self.alpha
        b[N-1] = 2 * self.gamma * delta_x

        # Create the array of x values that will be used in solvers
        x_array = self.x_values[1:]

        return A, b, x_array


    def construct_matrix(self):
        """
        Function to define the matrix A and the vector b for the boundary conditions based on the condition_type.

        Returns
        -------
        A : numpy.ndarray
            Matrix A. Size of A depends on the condition_type.
        b : numpy.ndarray
            Vector b. Size of b depends on the condition_type.
        x_array : numpy.ndarray
            Array of x values corresponding to A and b. Size of x_array depends on the condition_type.

        """
        if self.condition_type == 'Dirichlet':
            A, b, x_array = self.dirichlet_boundary_conditions()
            self.shape_ = 'N-1'
            self.shape = self.N-1
        
        elif self.condition_type == 'Neumann':
            A, b, x_array = self.neumann_boundary_conditions()
            self.shape_ = 'N'
            self.shape = self.N
        
        elif self.condition_type == 'Robin':
            A, b, x_array = self.robin_boundary_conditions()
            self.shape_ = 'N'
            self.shape = self.N
        
        else:
            raise ValueError('condition_type must be either Dirichlet or Neumann or Robin')
        
        if self.sparse:
            A = sparse.coo_matrix(A)
        
        return A, b, x_array


    def time_discretization(self, t_boundary, t_final, method, dt=None, C=None):
        """
        Function to discretize time from t_boundary to t_final. Either dt or C must be provided, and the other will be calculated.

        Parameters
        ----------
        t_boundary : float
            Initial time.
        t_final : float
            Final time.
        method : str
            Method to use for PDE. Determines if warning is raised for Courant number.
        dt : float, optional
            Time step. The default is None.
        C : float, optional
            Courant number. The default is None.

        Returns
        -------
        t : numpy.ndarray
            Array of time values.
        dt : float
            Time step.
        C : float
            Courant number.
        """
        # Work out the time step or Courant number
        if dt is not None:
            self.dt = dt
            self.C = self.D * dt / self.dx**2
        elif C is not None:
            self.C = C
            self.dt = C * self.dx**2 / self.D
        C = self.C
        dt = self.dt

        if method == 'Scipy Solver' or 'Explicit Euler':
            if C > 0.5:
                # Raise warning
                warnings.warn('C = D * dt / dx^2 = {} > 0.5. The solution may be unstable.'.format(C))
            
        N_time = ceil((t_final - t_boundary) / dt)
        t = dt * np.arange(N_time + 1) + t_boundary
        return t, dt, C
    
    
    def concatanate(self, u, type='ODE', t=None):
        """
        Function to concatanate the solution vector with the boundary conditions.

        Parameters
        ----------
        u : numpy.ndarray
            Solution vector.
        type : str, optional
            Type of problem. If type is 'ODE', then the singular boundary conditions are concatanated to the solution vector.
            If type is 'PDE', then the vector of boundary conditions is concatanated to the solution vector.
            The default is 'ODE'.
        t : numpy.ndarray, optional
            Array of time values. Must be provided if type is 'PDE'. The default is None.

        Returns
        -------
        numpy.ndarray
            Solution vector with boundary conditions concatanated.
        """
        # TODO: Add a check to make sure that u is consistent with type and t

        if self.condition_type == 'Dirichlet':
            if type == 'ODE':
                return np.concatenate(([self.alpha], u, [self.beta]))
            elif type == 'PDE':
                alpha_boundary = np.array([self.alpha] * len(t)).reshape(1, -1)
                beta_boundary = np.array([self.beta] * len(t)).reshape(1, -1)
                return np.concatenate((alpha_boundary, u, beta_boundary), axis=0)
            
        elif self.condition_type == 'Neumann' or self.condition_type == 'Robin':
            if type == 'ODE':
                return np.concatenate(([self.alpha], u))
            elif type == 'PDE':
                alpha_boundary = np.array([self.alpha] * len(t)).reshape(1, -1)
                return np.concatenate((alpha_boundary, u), axis=0)
        

        
