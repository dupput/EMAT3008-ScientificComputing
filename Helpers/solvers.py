import numpy as np

def euler_step(fun, t, y, h):
    """Perform one step of the Euler method.
    
    Parameters
    ----------
    fun : function
        Function f(t, y) to integrate.
    t : float
        Current value of t.
    y : float
        Current value of y.
    h : float
        Step size.
        
    Returns
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
    
    Parameters
    ----------
    fun : function
        Function f(t, y) to integrate.
    t : float
        Current value of t.
    y : float
        Current value of y.
    h : float
        Step size.
        
    Returns
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
    
    Parameters
    ----------
    fun : function
        Function f(t, y) to integrate.
    t : float
        Current value of t.
    y : float
        Current value of y.
    h : float
        Step size.
        
    Returns
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

    Parameters
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

    Returns
    -------
    t : array
        Array of t values.
    y : array
        Array of y values.
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

    while (t_max is None or t <= t_max) and (n_max is None or step <= n_max):
        t, y = method(fun, t, y, deltat_max)
        t_array.append(t)
        y_array.append(y)
        step += 1

    return np.array(t_array), np.array(y_array)


def shooting(initial_guess, ode, phase_function):
    '''
    Shooting method for solving boundary value problems. 
    
    parameters
    ----------
    initial_guess: list
        The initial guess should be a list of the form [y0, y1, ..., yn, T], where y0, y1, ..., yn
        are the initial conditions for the ODE and T is the final time.
    ode: function
        The ODE to be solved. The function should take the form f(t, y, *args), where t is the
        independent variable, y is the dependent variable, and *args are the parameters of the ODE.
    phase_function: function
        The phase function for the ODE. The function should take the form f(t, y, *args), where t is the
        independent variable, y is the dependent variable, and *args are the parameters of the ODE. The
        phase function should return the value of the derivative of the dependent variable with respect
        to the independent variable. Should an alternative value of 0 be desired, the value should be 
        incorporated into the phase functions return statement.

    returns
    -------
    X0: array
        The array of initial conditions for the ODE that satisfy the boundary conditions.
    T: float
        The final time period for the ODE.

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
    >>> initial_guess = [0.6, 0.8, 35]
    >>> X0, T = shooting(initial_guess, ode, phase_function)
    >>> print(X0, T)
    [0.8189692  0.16636172] 34.06960743834717
    '''

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

    sol = root(shooting_root, initial_guess)
    X0 = np.array(sol.x[:-1])
    T = sol.x[-1]

    return X0, T
