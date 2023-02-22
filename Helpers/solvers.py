from scipy.optimize import fsolve
import numpy as np

# Custom exceptions
class InputError(Exception):
    """Exception raised for errors in the input."""
    pass

class FunctionError(Exception):
    """Exception raised for errors in the function."""
    pass



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


def solve_to(fun, t0, y0, t_max=None, n_max=None, method='RK4', deltat_max=0.01, args=None):
    """Solve an ordinary differential equation.

    parameters
    ----------
    fun : function
        Function f(t, y) to integrate.
    t0 : float
        Initial value of t.
    y0 : float
        Initial value of y.
    t_max : float, optional
        Maximum value of t.
    n_max : int, optional
        Maximum number of steps.
    method : str, optional
        Integration method. Must be 'Euler' or 'RK4'.
    deltat_max : float, optional
        Maximum step size.
    args : tuple, optional
        Additional arguments to pass to fun. 

    returns
    -------
    t : array
        Array of t values.
    y : array
        Array of y values.

    examples
    --------
    >>> import numpy as np

    >>> def fun(t, y):
    ...     y_ = y[0]
    ...     x_ = y[1]
    ...     return np.array([x_, -y_])
    >>> t0 = 0
    >>> y0 = np.array([1, 0])
    >>> t_max = 20
    >>> deltat_max = 0.1

    >>> t, y = solve_to(fun, t0, y0, t_max=t_max, method='Euler', deltat_max=deltat_max)
    """
    METHODS = {'Euler': euler_step, 'RK4': rk4_step, 'Heun': heun_step}
    if method not in METHODS:
        raise ValueError('Invalid method: {}. Method must be one of {}.'.format(method, METHODS.keys()))
    else:
        method = METHODS[method]

    # Check that either t_max or n_max is given
    if t_max is None and n_max is None:
        raise ValueError('Either t_max or n_max must be given.')

    # Check args
    if args is not None:
        # Wrap the fun in lambdas to pass through additional parameters.
        try:
            _ = [*(args)]
        except TypeError as exp:
            suggestion_tuple = (
                "Supplied 'args' cannot be unpacked. Please supply `args`"
                f" as a tuple (e.g. `args=({args},)`)"
            )
            raise TypeError(suggestion_tuple) from exp

        fun = lambda t, x, fun=fun: fun(t, x, *args)

    t = t0
    y = y0
    t_array = [t]
    y_array = [y]
    step = 0

    while (t_max is None or t < t_max) and (n_max is None or step < n_max):
        t, y = method(fun, t, y, deltat_max)
        t_array.append(t)
        y_array.append(y)
        step += 1

        # Prevent overshooting t_max
        if t_max is not None and t + deltat_max > t_max:
            deltat_max = t_max - t

    return np.array(t_array), np.array(y_array)


def shooting(U0, ode, phase_function, atol=1e-8):
    '''
    Shooting method for solving boundary value problems. 
    
    parameters
    ----------
    U0: array
        The initial guess, U0 should be a list of the form [y0, y1, ..., yn, T], where y0, y1, ..., yn
        are the initial conditions for the ODE and T is the time period for the ODE.
    ode: function
        The ODE to be solved. The function should take the form f(t, y, *args), where t is the
        independent variable, y is the dependent variable, and *args are the parameters of the ODE.
    phase_function: function
        The phase function for the ODE. The function should take the same inputs as the ODE. The phase
        function is used to determine the differential at the start of the ODE and should be zero at
        the end of the ODE.
    atol: float, optional
        The absolute tolerance for the ODE solver. The default value is 1e-4.

    returns
    -------
    X0: array
        The array of initial conditions for the ODE that satisfy the boundary conditions.
    T: float
        The time period for the ODE.

    example
    -------
    >>> def ode(t, y, a=1, d=0.1, b=0.1):
    ...     x = y[0]
    ...     y = y[1]
    ...     dxdt = x*(1-x) - (a*x*y)/(d+x)
    ...     dydt = b*y*(1 - y/x)
    ...     return np.array([dxdt, dydt])
    >>> def phase_function(t, y):
    ...     dx, dy = ode(t, y)
    ...     return dx
    >>> U0 = [0.6, 0.8, 35]
    >>> X0, T = shooting(U0, ode, phase_function)
    '''
    # Check that the initial guess is of the correct form for the ODE
    try:
        solve_to(ode, 0, U0[:-1], n_max=1)
    except:
        raise InputError('The initial guess is not of the correct form. The initial guess should be a list of the form [y0, y1, ..., yn, T], where y0, y1, ..., yn are the initial conditions for the ODE and T is the time period.')

    try:
        phase_function(0, U0[:-1])
    except:
        raise FunctionError('The phase function is not of the correct form. The phase function should take the same inputs as the ODE.')


    # Set up function to optimize
    def shooting_root(initial_guess):
        T = initial_guess[-1]
        Y0 = np.array(initial_guess[:-1])

        # Solve the ODE
        t, y = solve_to(ode, 0, Y0, t_max=T)

        # Set up the conditions array
        num_vars = len(initial_guess)
        conditions = np.zeros(num_vars)
        # Dynamically fill in the conditions array
        for i in range(num_vars - 1):
            conditions[i] = y[-1,i] - Y0[i]        # X(T) - X(0) = 0

        # Final condition is the phase function
        phase_value = phase_function(0, Y0)
        conditions[-1] = phase_value               # dx/dt(T) = 0

        return conditions

    sol = fsolve(shooting_root, U0)
    X0 = np.array(sol[:-1])
    T = sol[-1]

    # Check that the solution is valid. 
    final_conditions = shooting_root(sol)
    if not np.allclose(final_conditions, np.zeros(len(sol)), atol=atol):
        message = '''The solution does not satisfy the boundary conditions. 
        The final conditions are: {}. 
        The final value of X0 and T is: {}, {}. 
        the initial guess or increasing the absolute tolerance.
        '''.format(final_conditions, X0, T)
        raise ValueError(message)
    
    if T <= 0:
        raise ValueError('The time period is not positive. Please try a different initial guess.')

    return X0, T

