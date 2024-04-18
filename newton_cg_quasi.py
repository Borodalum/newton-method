import time

from scipy.optimize import minimize


def newton_cg(f, grad_f, x0):
    return calc_time(newton_cg_method, (f, grad_f, x0))


def quasinewton(f, grad_f, x0):
    return calc_time(bfgs_method, (f, grad_f, x0))


def newton_cg_method(f, grad_f, x0):
    result = minimize(f, x0, method='Newton-CG', jac=grad_f)
    return result.x, result.nit


def bfgs_method(f, grad_f, x0):
    result = minimize(f, x0, method='BFGS', jac=grad_f)
    return result.x, result.nit


def calc_time(func, args):
    start = time.time()
    x, iterations = func(*args)
    end = time.time() - start
    return x, iterations, end
