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


class ODE:
    """
    Class to solve ordinary differential equations.
    
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
        Integration method. Can be 'Euler', 'RK4' or 'Heun'.
    deltat_max : float, optional
        Maximum step size.
    function_args : tuple, optional
        Additional arguments to pass to fun. 
    """
    METHODS = {'Euler': euler_step, 'RK4': rk4_step, 'Heun': heun_step}

    def __init__(self, fun, t0, y0, t_max=None, n_max=None, method='RK4', deltat_max=0.01, function_args=None):
        self.fun = fun
        self.t0 = t0
        self.y0 = y0
        self.t_max = t_max
        self.n_max = n_max
        self.method = method
        self.deltat_max = deltat_max
        self.function_args = function_args

        # Check args
        if self.function_args is not None:
            # Wrap the fun in lambdas to pass through additional parameters.
            try:
                _ = [*self.function_args]
            except TypeError as exp:
                suggestion_tuple = (
                    "Supplied 'args' cannot be unpacked. Please supply `args`"
                    f" as a tuple (e.g. `args=({function_args},)`)"
                )
                raise TypeError(suggestion_tuple) from exp

            self.fun = lambda t, x, fun=self.fun: fun(t, x, *self.function_args)

        # Check method
        if self.method not in self.METHODS:
            raise ValueError('Invalid method: {}. Method must be one of {}.'.format(self.method, self.METHODS.keys()))
        else:
            self.method = self.METHODS[self.method]

        # Check that either t_max or n_max is given
        if self.t_max is None and self.n_max is None:
            raise ValueError('Either t_max or n_max must be given.')


    def solve_to(self):
        """
        Solve the ODE.

        Returns
        -------
        t_array : array <float>
            Array of t values.
        y_array : array <float>
            Array of y values.
        
        """
        t = self.t0
        y = self.y0
        t_array = [t]
        y_array = [y]
        step = 0

        while (self.t_max is None or t <= self.t_max) and (self.n_max is None or step <= self.n_max):
            t, y = self.method(self.fun, t, y, self.deltat_max)
            t_array.append(t)
            y_array.append(y)
            step += 1

        return np.array(t_array), np.array(y_array)
        

    def shooting(self, phase_function, phase_condition, y_step=0.01, yn=0):
        """
        This function uses the shooting method to solve a boundary value problem. The phase condition is evaluated at the
        end of the integration and the initial condition is adjusted until the phase condition is met.

        The recommended phase conditions is to set the phase function as the differential and the phase condition as 0.

        Parameters
        ----------
        phase_function : function
            Function to evaluate the phase condition. Must be of the form f(t, y).
        phase_condition : float
            Target value of the phase condition.
        y_step : float, optional
            Rate of change of y. Value is a multiplier of the difference between the target and the current value.
        yn : int, optional
            Index value of y to vary.

        Returns
        -------
        t : array
            Array of t values.
        y : array
            Array of y values.
        """
        # Initial guess for y0
        t, y = self.solve_to()

        # Evaluate phase condition
        deltas = phase_function(t, y)
        
        # Check if phase function is correct dimension
        if len(deltas) != len(self.y0):
            raise ValueError('Phase function must return an array of the same dimension as y0.')
        else:
            dy = deltas[yn]

        # Extract initial conditions and parameter to vary
        v0 = self.y0[yn]

        # Iterate until phase condition is met
        iter = 0
        while np.round(dy[-1], 2) != 0:
            # Increse v0 in the direction of dy
            v0 += y_step * np.sign(dy[-1])

            # Update y0 in object
            self.y0[yn] = v0
            
            # Solve ODE
            t, y = self.solve_to()
            deltas = phase_function(t, y)
            dy = deltas[yn]

        print('Final guess: y0 = {}, dy/dt = {}'.format(self.y0, dy[-1]))

        return t, y
    

