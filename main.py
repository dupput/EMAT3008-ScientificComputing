import numpy as np
import matplotlib.pyplot as plt


def Eulers(fun, t0, y0, tn, h=0.01, atol=1e-6):
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
    n = (tn - t0)/h

    for _ in range(int(n)):
        m = fun(t0, y0)
        y0 = y0 + h*m
        t0 = t0 + h
    return y0


fun = lambda t, y: 2 - np.exp(-4*t) - 2*y
t0 = 0
y0 = 1

y = Eulers(fun, t0, y0, 0.3)
print(y)