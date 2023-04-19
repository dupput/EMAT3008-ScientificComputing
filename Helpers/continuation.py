import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np

def continuation(ode, x0, param_current, variation_param, step_size, max_steps, method='parameter continuation'):
    """
    Performs natural parameter continuation on the given ODE.

    Parameters
    ----------
    ode : function
        ODE to be solved.
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

            t, x = ode(0, X_next, b_next)

            condition1 = x

            if method == 'arc-length continuation' and len(x_sol) >= 2:
                condition2 = (X_current - X_next) * delta_X + (param_current[variation_param] - b_next) * delta_param
            else:
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
        if len(x_sol) >= 5:
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

ode = lambda t, x, c: [t, x**3 - x + c]
x0 = 1
par0 = [-3]
vary_par = 0
step_size = 0.01
max_steps = 2000

x, par, dpars = continuation(ode, x0, par0, vary_par, step_size, max_steps)

plt.plot(par, x, '.-', label='scipy.optimize.fsolve', markersize=2)
plt.xlabel('Parameter')
plt.ylabel('x')
plt.grid()
plt.show()

from Helpers.solvers import shooting, solve_to

# import matplotlib.pyplot as plt
# from scipy.optimize import fsolve
# import numpy as np

# def ODE(t, y, beta, sigma):
#     u1 = y[0]
#     u2 = y[1]
#     du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
#     du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
#     return np.array([du1dt, du2dt])

# def phase_function(t, y, beta, sigma):
#     return ODE(t, y, beta, sigma)[0]


# X_current = np.array([1, 1, 6.28])
# par0 = [2, -1]
# variation_param = 0
# param_step_size = -0.1
# X_step_size = 0.01
# max_steps = 20

# method = 'parameter continuation'
# # method = 'arc-length continuation'

# X_solutions = []
# param_solutions = []
# t0 = 0

# step = 0
# param_current = par0
# num_vars = len(X_current)-1

# while step <= max_steps:
#     try:
#         X0, T = shooting(X_current, ODE, phase_function, function_args=param_current)
#         converged = True
#     except:
#         converged = False
    
    
#     if converged:
#         X_solutions.append(X0)
#         param_solutions.append(param_current[variation_param])
    

#     if method == 'parameter continuation' or len(X_solutions) < 2:
#         param_current[variation_param] += param_step_size
#         X_current += X_step_size


#     elif method == 'arc-length continuation':
#         # Predict next solution. u = u0 + delta_u
        
#         def func(parameter):
#             # Construct u0 array
#             u0 = np.array([])
#             for i in range(num_vars):
#                 u0 = np.append(u0, X_current[i])
#             u0 = np.append(u0, parameter)

#             # Calculate delta_X and delta param
#             delta_X = X_solutions[-1] - X_solutions[-2]
#             delta_param = param_solutions[-1] - param_solutions[-2]

#             # Construct delta_u array
#             delta_u = np.array([])
#             for i in range(num_vars):
#                 delta_u = np.append(delta_u, delta_X[i])
#             delta_u = np.append(delta_u, delta_param)

#             # Construct u array
#             u = u0 + delta_u

#             condition = np.dot(u0 - u, delta_u)

#             return condition

#         fs = fsolve(func, param_current[variation_param], full_output=True)
        
#         converged = fs[2] == 1
#         if converged:
#             param_current[variation_param] = fs[0]
#             X_current = X_solutions[-1]
#         else:
#             param_current[variation_param] += param_step_size
#             X_current += X_step_size
#     step += 1

