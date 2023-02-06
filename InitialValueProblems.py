import numpy as np
import matplotlib.pyplot as plt
import time

from Helpers.solvers import solve_to

if __name__ == '__main__':    
    # Define ODE
    fun = lambda t, y: y
    fun_analytical = lambda t: np.exp(t)
    t0 = 0
    y0 = 1
    t_max = 1

    # -----------------------------------------------------------------------------------------
    # COMPARISON OF EFFECT OF STEP SIZE
    deltas = [10, 1, 0.1, 0.01, 0.001]
    methods = ['Euler', 'RK4', 'Heun']

    for method in methods:
        error = []

        for timestep in deltas:
            t_, y_ = solve_to(fun, t0, y0, t_max=t_max, method=method, deltat_max=timestep)
            error.append(np.abs(y_[-1] - fun_analytical(t_max)))

        # plt.loglog(deltas, error, '.-', label=method) 

    # plt.xlabel('Timestep')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.show()

    # -----------------------------------------------------------------------------------------
    # COMPARISON OF TIME TAKEN FOR SIMILAR ACCURACY

    # Find the timestep that gives the same error between the Euler and RK4 methods
        # baseline = 0.05
        # t_, y_ = solve_to(fun, t0, y0, t_max=t_max, method='RK4', deltat_max=baseline)
        # error = np.abs(y_[-1] - fun_analytical(t_max))

        # euler_error = np.Inf
        # while euler_error > error:
        #     print('Time step: {}'.format(baseline))
        #     print('RK4 error: {}'.format(error))
        #     print('Euler error: {}'.format(euler_error))
        #     baseline /= 10
        #     t_, y_ = solve_to(fun, t0, y0, t_max=t_max, method='Euler', deltat_max=baseline)
        #     euler_error = np.abs(y_[-1] - fun_analytical(t_max))

        # print('The timestep that gives the same error between the methods is: {}'.format(baseline))
    # This loop results in timestep of 5e-07
    # start = time.time()
    # t_, y_ = solve_to(fun, t0, y0, t_max=t_max, method='Euler', deltat_max=5e-07)
    # end = time.time()
    # euler_error = np.abs(y_[-1] - fun_analytical(t_max))    
    # euler_time = end - start

    # start = time.time()
    # t_, y_ = solve_to(fun, t0, y0, t_max=t_max, method='RK4', deltat_max=0.05)
    # end = time.time()
    # rk4_time = end - start
    # rk4_error = np.abs(y_[-1] - fun_analytical(t_max))

    # print('Euler error: {}, time taken: {}'.format(euler_error, euler_time))
    # print('RK4 error: {}, time taken: {}'.format(rk4_error, rk4_time))

    # -----------------------------------------------------------------------------------------
    # SYSTEM OF ODES

    # Define ODE
    # d^2y/dt^2 = -y ==> dy/dt = v, dv/dt = -u
    def fun(t, y):
        y_ = y[0]
        x_ = y[1]
        return np.array([x_, -y_])

    def fun_analytical(t):
        return np.cos(t)     
    
    t0 = 0
    y0 = np.array([1, 0])
    t_max = 20
    deltat_max = 0.1

    # Solve ODE
    t_euler, y_euler = solve_to(fun, t0, y0, t_max=t_max, method='Euler', deltat_max=deltat_max)
    t_rk4, y_rk4 = solve_to(fun, t0, y0, t_max=t_max, method='RK4', deltat_max=deltat_max)
    # t_heun, y_heun = solve_to(fun, t0, y0, t_max=t_max, method='Heun', deltat_max=deltat_max)

    # Analytical solution
    y_a = fun_analytical(t_rk4)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plot results: y vs t
    ax1.plot(t_rk4, y_rk4[:, 0], label='RK4')
    ax1.plot(t_rk4, y_a, '--', label='Analytical')
    ax1.set(xlabel='t', ylabel='y')
    ax1.legend()

    # # Plot results: y vs y'
    ax2.plot(y_euler[:, 0], y_euler[:, 1], label='Euler')
    ax2.plot(y_rk4[:, 0], y_rk4[:, 1], label='RK4')
    ax2.set(xlabel='y', ylabel='dy/dt')
    ax2.legend()
    plt.show()


