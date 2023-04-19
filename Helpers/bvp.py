from math import ceil 
import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
import warnings

class InputError(Exception):
    """Exception raised for errors in the input."""
    pass

class FunctionError(Exception):
    """Exception raised for errors in the function."""
    pass

class BVP:
    """
    Class for solving boundary value problems.

    Functions inside the class have been split into distinct sections for clarity. The sections are:
    - Check functions
    - Discretisation functions
    - Boundary conditions functions
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
    """
    def __init__(self, a, b, N, alpha, beta=None, delta=None, gamma=None, condition_type=None, 
                 q_fun=None, f_fun=None, D=1):
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

        # Check that the boundary conditions are valid
        self.check_boundary_validity()

        # Check q_fun and f_fun
        self.check_functions()

        # Create the grid
        self.grid_discretisation()

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
                raise InputError('alpha and beta must be specified for Dirichlet boundary conditions')

        elif self.condition_type == 'Neumann':
            if self.alpha is None or self.delta is None:
                raise InputError('alpha and delta must be specified for Neumann boundary conditions')
            
        elif self.condition_type == 'Robin':
            if self.alpha is None or self.delta is None or self.gamma is None:
                raise InputError('alpha, delta and gamma must be specified for Robin boundary conditions')
            
        else:
            raise ValueError('condition_type must be either Dirichlet or Neumann or Robin')


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


    # -------------------------------- Discretisation functions --------------------------------
    def grid_discretisation(self):
        """
        Create the grid. The grid is a numpy array of N+1 points between a and b.
 
        Returns
        -------
        None.
        """
        # Check if a and b are numbers
        if not isinstance(self.a, (int, float)) or not isinstance(self.b, (int, float)):
            raise InputError('a and b must be numbers. a = {}, b = {}'.format(self.a, self.b))

        # Check if N is an integer
        if not isinstance(self.N, int) or self.N < 0:
            raise InputError('N must be an positive integer. N = {}'.format(self.N))

        self.x_values = np.linspace(self.a, self.b, self.N + 1)
        self.dx = (self.b - self.a) / self.N


    def time_discretization(self, t_boundary, t_final, dt=None, C=None):
        """
        Function to discretize time from t_boundary to t_final. Either dt or C must be provided, and the other will be calculated.

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
        if dt is None and C is None:
            raise InputError('Either dt or C must be provided')
        elif dt is not None and C is not None:
            raise InputError('Only one of dt or C must be provided')
        elif dt is not None:
            self.dt = dt
            self.C = self.D * dt / self.dx**2
        elif C is not None:
            self.C = C
            self.dt = C * self.dx**2 / self.D
        C = self.C
        dt = self.dt

        if C > 0.5:
            # Raise warning
            warnings.warn('C = D * dt / dx^2 = {} > 0.5. The solution may be unstable.'.format(C))
        
        N_time = ceil((t_final - t_boundary) / dt)
        t = dt * np.arange(N_time + 1) + t_boundary
        return t, dt, C


    # -------------------------------- Boundary condition functions --------------------------------
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
        
        return A, b, x_array

    
    # -------------------------- ODE Solvers -------------------------- #
    def solve_matrices(self, A, b, u_array=None, x_array=None):
        """
        Function to solve the linear system of equations Au = -b - q(x, u) * dx^2.
        The function q(x, u) is defined by the user and must be linear in u.

        Parameters
        ----------
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
        if self.q_fun is not None:
            if x_array is None or u_array is None:
                raise ValueError('x_array and u_array must be provided if q_fun is provided')
            else:
                b = b + self.q_fun(x_array, u_array) * self.dx**2

        try:        
            # Solve the linear system
            u = np.linalg.solve(A, -b)
            success = True

            return u, success
        
        except:
            success = False
            return None, success
        

    def solve_thomas_algorithm(self, A, b, u_array=None, x_array=None):
        """
        Function to solve the linear system of equations Au = -b - q(x, u) * dx^2.
        The function q(x, u) is defined by the user and must be linear in u.
        The function uses the Thomas algorithm to solve the linear system of equations:

            a_i * u_{i-1} + b_i * u_i + c_i * u_{i+1} = d_i

        where
            d_i = -b_i - q(x_i, u_i) * dx^2

        Parameters
        ----------
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
        if self.q_fun is not None:
            if x_array is None or u_array is None:
                raise ValueError('x_array and u_array must be provided if q_fun is provided')
            else:
                d = -(b + self.q_fun(x_array, u_array) * self.dx**2)
        else:
            d = -b

        try:        
            # define the vectors a, b and c
            a = A.diagonal(-1).copy()
            b = A.diagonal().copy()
            c = A.diagonal(1).copy()

            # Forward sweep
            for i in range(1, self.shape):
                m = a[i-1]/b[i-1]
                b[i] = b[i] - m * c[i-1]
                d[i] = d[i] - m * d[i-1]

            # Backward sweep
            u = np.zeros(self.shape)
            u[-1] = d[-1]/b[-1]
            for i in range(self.shape-2, -1, -1):
                u[i] = (d[i] - c[i] * u[i+1])/b[i]

            success = True
            return u, success
        
        except Exception as e:
            success = False
            return u_array, success


    def solve_nonlinear(self, A, b, u_array, x_array=None):
        """
        Function to solve the nonlinear system of equations Au = -b - q(x, u) * dx^2.
        The function q(x, u) is defined by the user and can be nonlinear in u.

        Parameters
        ----------
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
        
        if self.q_fun is None:
            if x_array is None:
                raise ValueError('x_array and u_array must be provided if q_fun is not provided')
            else:
                func = lambda u_array: self.D * A @ u_array + b
        else:
            func = lambda u_array: self.D * A @ u_array + b + self.q_fun(x_array, u_array) * self.dx**2
        
        sol = root(func, u_array)
        u = sol.x
        
        return u, sol.success
    

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
        # Method check
        METHODS = {'linear': self.solve_matrices, 'tridiagonal': self.solve_thomas_algorithm,
                   'nonlinear': self.solve_nonlinear}
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
            raise InputError(message)

        # Solve the boundary value problem
        u, success = method(A, b, u_array=u_array, x_array=x_array)

        if not success:
            warnings.warn('The solver was not successful. The solution may not be accurate.')

        # Concatanate the solution with the boundary conditions
        u = self.concatanate(u, type='ODE')

        return u, success
    

    # -------------------------- PDE Solvers -------------------------- #
    def scipy_solver(self, t):
        """
        Function to solve the PDE using Scipy's solve_ivp.

        Parameters
        ----------
        t : numpy.ndarray
            Array of time values.
        t_boundary : float
            Initial time.

        Returns
        -------
        u : numpy.ndarray
            Solution to the PDE.
        """
        A, b, x_array = self.construct_matrix()
        
        u_boundary = self.f_fun(x_array, t[0])

        if self.q_fun is None:
            def PDE(t, u, D, A, b):
                return D / self.dx ** 2 * (A @ u + b)
        else:
            def PDE(t, u, D, A, b):
                return D / self.dx ** 2 * (A @ u + b) + self.q_fun(x_array, u)  

        sol = solve_ivp(PDE, (t[0], t[-1]), u_boundary, method='RK45', t_eval=t, args=(self.D, A, b))

        y = sol.y
        t = sol.t

        u = self.concatanate(y, type='PDE', t=t)
        return u
    

    def explicit_euler(self, t):
        """
        Function to solve the PDE using the explicit Euler method.

        Parameters
        ----------
        t : numpy.ndarray
            Array of time values.

        Returns
        -------
        solution : numpy.ndarray
            Solution to the PDE.
        """
        u = np.zeros((len(self.x_values), len(t)))
        u[:, 0] = self.f_fun(self.x_values, t[0])
        u[0, :] = self.alpha
        u[-1, :] = self.beta

        # Loop over time
        for ti in range(0, len(t)-1):
            for xi in range(0, self.N):
                if xi == 0:
                    # u[0, ti+1] = u[0, ti] + self.C * (self.alpha - 2 * u[0, ti] + u[1, ti])
                    u[xi, ti+1] = self.alpha
                elif xi > 0 and xi < self.N:
                    u[xi, ti+1] = u[xi, ti] + self.C * (u[xi-1, ti] - 2 * u[xi, ti] + u[xi+1, ti])
                else:
                    u[xi, ti+1] = u[xi, ti] + self.C * (self.beta - 2 * u[xi, ti] + u[xi-1, ti])

        return u
    

    def implicit_euler(self, t):
        """
        Function to solve the PDE using the implicit Euler method.

        Parameters
        ----------
        t : numpy.ndarray
            Array of time values.

        Returns
        -------
        solution : numpy.ndarray
            Solution to the PDE.
        """
        A, b, x_array = self.construct_matrix()

        I = np.eye(self.N-1)

        lhs = I - self.C * A
        rhs = self.C * b

        u = np.zeros((len(x_array), len(t)))
        u[:, 0] = self.f_fun(x_array, t[0])

        for ti in range(0, len(t)-1):
            u[:, ti+1] = np.linalg.solve(lhs, u[:, ti] + rhs)

        return u
    

    def crank_nicolson(self, t):
        """
        Function to solve the PDE using the Crank-Nicolson method.

        Parameters
        ----------
        t : numpy.ndarray
            Array of time values.

        Returns
        -------
        solution : numpy.ndarray
            Solution to the PDE.
        """
        A, b, x_array = self.construct_matrix()

        I = np.eye(self.N-1)

        lhs = I - self.C * A / 2
        rhs = I + self.C * A / 2

        u = np.zeros((len(x_array), len(t)))
        u[:, 0] = self.f_fun(x_array, t[0])

        for ti in range(0, len(t)-1):
            u[:, ti+1] = np.linalg.solve(lhs, rhs @ u[:, ti] + self.C * b)

        return u
    
    
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
        
    
    





if __name__ == '__main__':
    a = 0
    b = 1
    N = 10
    alpha = 0
    beta = 0
    condition_type = 'Dirichlet'
    q_fun = lambda x, u: 1

    bvp = BVP(a, b, N, alpha, beta, condition_type=condition_type, q_fun=q_fun)

    # u_DD, success = bvp.solve_ODE()
    u_tridiag, success = bvp.solve_ODE(method='tridiagonal')
    u_linear, success = bvp.solve_ODE(method='linear')
    u_nonlinear, success = bvp.solve_ODE(method='nonlinear')

    def u_analytic(x):
        return -1/2 * (x - a) * (x - b) + ((beta - alpha) / (b - a)) * (x - a) + alpha

    x_values = bvp.x_values
    u_analytic = u_analytic(x_values)

    # Assert that the solutions are the same
    assert np.allclose(u_tridiag, u_linear, u_nonlinear, u_analytic)