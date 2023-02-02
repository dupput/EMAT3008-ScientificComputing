
from solvers import Solver

class Eulers(Solver):
    """
    Integrate a system of ordinary differential equations.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
    t0 : array_like
        A sequence of time points for which to solve for y. The initial value
        point should be the first element of this sequence.
    y0 : array_like
        Initial condition on y (can be a vector).
    tn : int
        Desired value of t at the end of the integration.
    h : float
        Step size
    atol : float
        Absolute tolerance. If the norm of the difference between two
        consecutive steps is less than this value, the integration is
        terminated.
    """
    def __init__(self, fun, t0, y0, tn, h=0.01, atol=1e-6, filename='Euler.png'):
        self.fun = fun
        self.t0 = t0
        self.y0 = y0
        self.tn = tn
        self.h = h
        self.atol = atol
        self.filename = filename

    def solve(self, plot=False):
        t = self.t0
        y = self.y0
        h = self.h
        fun = self.fun

        t_array = [t]
        y_array = [y]
        while t < self.tn:
            m = fun(t, y)
            t = t + h
            y = y + h*m

            t_array.append(t)
            y_array.append(y)
        
        if plot:
            self.plot_solution(t_array, y_array, self.filename)
        
        return y
