import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np

def simple_continuation(function, x0, param_current, variation_param, step_size, max_steps, method='parameter continuation'):
    """
    Performs natural parameter continuation for a function of the form f(t, x, b).

    Parameters
    ----------
    func : function of the form f(t, x, b)
        Function to be undergo continuation.
    x0 : float
        Initial condition for the ODE.
    param_current : list
        List of parameters for the ODE.
    variation_param : int
        Index of the parameter to be varied.
    step_size : float
        Step size for the continuation.
    max_steps : int
        Maximum number of steps to take.
    method : str, optional
        Method to use for continuation. The default is 'parameter continuation'.

    Returns
    -------
    x_sol : list
        List of solutions.
    par_sol : list
        List of parameters.
    dpars : list
        List of parameter steps.

    Examples
    --------
    >>> func = lambda t, x, c: [t, x**3 - x + c]
    >>> x0 = 1
    >>> par0 = [-3]
    >>> vary_par = 0
    >>> step_size = 0.01
    >>> max_steps = 2000
    >>> x, par, dpars = simple_continuation(func, x0, par0, vary_par, step_size, max_steps)
    """        

    param_current[variation_param] = param_current[variation_param]
    guess = np.array([x0, param_current[variation_param]])
    x_sol = []
    par_sol = []
    dpars = []

    step = 0
    while step <= max_steps:
        def func(X0):

            X_next, b_next = X0

            t, x = function(0, X_next, b_next)

            condition1 = x

            condition2 = b_next - param_current[variation_param]

            return np.array([condition1, condition2])
        
        fs = fsolve(func, guess, full_output=True)

        # Extract the solution and check if the solver converged
        sol = fs[0]
        X_current, param_current[variation_param] = sol
        converged = fs[2] == 1

        if converged:
            x_sol.append(X_current)
            par_sol.append(param_current[variation_param])

        # Update the initial guess for the next step based on previous two solutions
        if len(x_sol) >= 3 and method == 'arc-length continuation':
            # Find delta
            delta_param = param_current[variation_param] - param_previous
            dpars.append(delta_param)
            # Update b_i and b_i-1
            param_previous = param_current[variation_param]
            param_current[variation_param] += delta_param

            delta_X = X_current - X_previous
            X_previous = X_current
            X_current += delta_X - step_size


        # Update next step from arbitrary step size
        else:
            X_previous = X_current
            X_current -= step_size

            param_previous = param_current[variation_param]
            param_current[variation_param] += step_size


        guess = np.array([X_current, param_current[variation_param]])
        step += 1

    return x_sol, par_sol, dpars

if __name__ == '__main__':
    func = lambda t, x, c: [t, x**3 - x + c]
    x0 = 1
    par0 = [-3]
    vary_par = 0
    step_size = 0.01
    max_steps = 2000

    x, par, dpars = simple_continuation(func, x0, par0, vary_par, step_size, max_steps)

    plt.plot(par, x, '.-', label='scipy.optimize.fsolve', markersize=2)
    plt.xlabel('Parameter')
    plt.ylabel('x')
    plt.grid()
    plt.show()
