import numpy as np

class BVP:
    def __init__(self, a, b, N, alpha, beta=None, delta=None, gamma=None, condition_type=None):
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.N = N
        self.condition_type = condition_type

        # Check that the boundary conditions are valid
        self.check_boundary_validity()

        # Create the grid
        self.grid()

    
    def check_boundary_validity(self):
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
            

    def grid(self):
        self.x_values = np.linspace(self.a, self.b, self.N + 1)
        self.delta_x = (self.b - self.a) / self.N


    def dirichlet_boundary_conditions(self):
        N = self.N
        delta_x = self.delta_x

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

        return A, b
    
    def neumann_boundary_conditions(self):
        N = self.N
        delta_x = self.delta_x

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
        A[N-1, N-1] = -2 * (1 + self.delta * delta_x)
        A[N-1, N-2] = 2
        b[0] = self.alpha
        b[N-1] = 2 * self.delta * delta_x

        return A, b
    
    def robin_boundary_conditions(self):
        N = self.N
        delta_x = self.delta_x

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
        A[N-1, N-1] = -2
        A[N-1, N-2] = 2
        b[0] = self.alpha
        b[N-1] = 2 * self.gamma * delta_x

        return A, b
    
    
    def boundary_conditions(self):
        if self.condition_type == 'Dirichlet':
            return self.dirichlet_boundary_conditions()
        
        elif self.condition_type == 'Neumann':
            return self.neumann_boundary_conditions()
        
        elif self.condition_type == 'Robin':
            return self.robin_boundary_conditions()
    

