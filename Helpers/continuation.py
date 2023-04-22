import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np

# Create NotImplemetedError for arc-length continuation
class NotImplementedError(Exception):
    pass


def simple_continuation(function, x0, paremeter_current, variation_parameter, step_size, max_steps, method='parameter continuation'):
    """
    Performs natural parameter continuation for a function of the form f(t, x, b).

    Parameters
    ----------
    func : function of the form f(t, x, b)
        Function to be undergo continuation.
    x0 : float
        Initial condition for the ODE.
    paremeter_current : list
        List of parameters for the ODE.
    variation_parameter : int
        Index of the parameter to be varied.
    step_size : float
        Step size for the continuation.
    max_steps : int
        Maximum number of steps to take.
    method : str, optional
        Method to use for continuation. The default is 'parameter continuation'.

    Returns
    -------
    x_solutions : list
        List of solutions.
    paremeter_solutions : list
        List of parameters.
    dpars : list
        List of parameter steps.

    Examples
    --------
    >>> func = lambda t, x, c: [t, x**3 - x + c]
    >>> x0 = 1
    >>> par0 = [-3]
    >>> variation_parameter = 0
    >>> step_size = 0.01
    >>> max_steps = 2000
    >>> x, parameters, delta_parameters = simple_continuation(func, x0, par0, variation_parameter, step_size, max_steps)
    """        

    paremeter_current[variation_parameter] = paremeter_current[variation_parameter]
    guess = np.array([x0, paremeter_current[variation_parameter]])
    x_solutions = []
    paremeter_solutions = []
    dpars = []

    step = 0
    while step <= max_steps:
        def func(X0):

            X_next, b_next = X0

            t, x = function(0, X_next, b_next)

            condition1 = x

            condition2 = b_next - paremeter_current[variation_parameter]

            return np.array([condition1, condition2])
        
        fs = fsolve(func, guess, full_output=True)

        # Extract the solution and check if the solver converged
        sol = fs[0]
        X_current, paremeter_current[variation_parameter] = sol
        converged = fs[2] == 1

        if converged:
            x_solutions.append(X_current)
            paremeter_solutions.append(paremeter_current[variation_parameter])

        # Update the initial guess for the next step based on previous two solutions
        if len(x_solutions) >= 3 and method == 'arc-length continuation':
            # Find delta
            delta_param = paremeter_current[variation_parameter] - param_previous
            dpars.append(delta_param)
            # Update b_i and b_i-1
            param_previous = paremeter_current[variation_parameter]
            paremeter_current[variation_parameter] += delta_param

            delta_X = X_current - X_previous
            X_previous = X_current
            X_current += delta_X - step_size


        # Update next step from arbitrary step size
        else:
            X_previous = X_current
            X_current -= step_size

            param_previous = paremeter_current[variation_parameter]
            paremeter_current[variation_parameter] += step_size


        guess = np.array([X_current, paremeter_current[variation_parameter]])
        step += 1

    return x_solutions, paremeter_solutions

def bvp_continuation(bvp, C_start, C_end, N_C, q_base, method='natural_parameter_estimation'):
    """
    Performs continuation for a boundary value problem.

    Parameters
    ----------
    bvp : BVP object
        Boundary value problem to be solved.
    C_start : float
        Starting value of C.
    C_end : float
        Ending value of C.
    N_C : int
        Number of steps to take.
    q_base : function
        Base q function.
    method : str, optional
        Method to use for continuation. The default is 'natural_parameter_estimation'.

    Returns
    -------
    Cs : list
        List of values of C.
    all_u : list
        List of solutions.

    Examples
    --------
    >>> a = 0
    >>> b = 1
    >>> N = 10
    >>> alpha = 0
    >>> beta = 0
    >>> D = 1.0
    >>> C = 0
    >>> def q_base(x, u, C=C):
            return C*np.exp(u)
    >>> q_fun = lambda x, u: q_base(x, u, C)

    >>> bvp = BVP(a, b, N, alpha, beta, condition_type='Dirichlet', q_fun=q_fun, D=D)
    >>> Cs, all_u = bvp_continuation(bvp, 0, 4, 200, q_base)
    """

    # Validate input parameters
    if N_C <= 0:
        raise ValueError('N_C must be a positive integer.')
    if C_start >= C_end:
        raise ValueError('C_start must be less than C_end.')

    # Choose method for continuation
    if method == 'natural_parameter_estimation':
        Cs = np.linspace(C_start, C_end, N_C)
    elif method == 'arc-length continuation':
        raise NotImplementedError('Arc-length continuation is not yet implemented.')
    else:
        raise ValueError('Method must be either "natural_parameter_estimation" or "arc-length continuation".')

    # Initialize arrays to determine sizes
    bvp.construct_matrix()
    u_array = np.ones(bvp.shape)
    all_u = np.zeros((N_C, bvp.N+1))

    # Perform continuation for each value of C
    for C in Cs:
        # Redefine the q function with the new value of C in terms of q
        q = lambda x, u: q_base(x, u, C)
        bvp.q_fun = q

        # Initialize the guess for the solution
        u_guess = np.zeros(bvp.N-1)
        
        # Solve the ODE with the current value of C
        u_array, success = bvp.solve_ODE(u_guess, 'nonlinear')

        # Store the solution if successful
        if success:
            all_u[Cs == C, :] = u_array

    return Cs, all_u


if __name__ == '__main__':
    from bvp import BVP
    import numpy as np
    import plotly.graph_objects as go
    a = 0
    b = 1
    N = 10
    alpha = 0
    beta = 0
    D = 1.0
    C = 0
    def q_base(x, u, C=C):
        return C*np.exp(u)
    q_fun = lambda x, u: q_base(x, u, C)

    bvp = BVP(a, b, N, alpha, beta, condition_type='Dirichlet', q_fun=q_fun, D=D)
    Cs, all_u = bvp_continuation(bvp, 0, 4, 200, q_base)
    max_u = np.max(all_u, axis=1)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Cs[max_u != 0], max_u[max_u != 0])
    ax.set_xlabel('C')
    ax.set_ylabel('Maximum value of u(x)')
    plt.show()
