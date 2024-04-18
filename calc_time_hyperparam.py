import time


def calc_time(func, arguments):
    start = time.time()
    x, iterations = func(*arguments)
    end = time.time() - start
    return x, iterations, end
