
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
        

def solve_ode(fun, t0, y0, t_max=None, n_max=None, method='Euler', deltat_max=0.01, filename=None):
    """Solve an ordinary differential equation.

    Parameters
    ----------
    fun : function
        Function f(t, y) to integrate.
    t0 : float
        Initial value of t.
    y0 : float
        Initial value of y.
    t_max : float
        Maximum value of t.
    n_max : int
        Maximum number of steps.
    method : str
        Integration method. Must be 'Euler' or 'RK4'.
    deltat_max : float
        Maximum step size.
    filename : str
        Name of file to save plot to. If None, plot is shown but not saved.

    Returns
    -------
    t : array
        Array of t values.
    y : array
        Array of y values.
    """
    METHODS = {'Euler': euler_step, 'RK4': rk4_step}
    if method not in METHODS:
        raise ValueError('Invalid method: {}. Method must be one of {}.'.format(method, METHODS.keys()))
    else:
        method = METHODS[method]

    # Check that either t_max or n_max is given
    if t_max is None and n_max is None:
        raise ValueError('Either t_max or n_max must be given.')

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

    # plt.plot(t_array, y_array)
    # plt.xlabel('t')
    # plt.ylabel('y')
    # if filename is not None:
    #     plt.savefig(filename)
    #     plt.close()
    # else:
    #     plt.show()
