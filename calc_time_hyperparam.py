import time


def calc_time(func, arguments):
    start = time.time()
    x, iterations, all_points = func(*arguments)
    end = time.time() - start
    return x, iterations, end, all_points
