%%
a = 0
b = 1
N = 10
alpha = 0
beta = 10
condition_type = 'Dirichlet'
q_fun = lambda x, u: 0

bvp = BVP(a, b, N, alpha, beta, condition_type=condition_type, q_fun=q_fun)

u = bvp.solve_ODE()

%%