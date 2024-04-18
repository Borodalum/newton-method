import numpy as np

from newton_cg_quasi import newton_cg, quasinewton
from newton_method_const import newton_with_constant_learning_rate
from newton_method_search import newton_with_search_learning_rate
from newton_wolfe import newton_with_wolfe_search
from plotting import plot_graphs

# Функция Розенброка
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


# Градиент функции Розенброка
def grad_rosenbrock(x):
    return np.array([400 * x[0] ** 3 - 400 * x[0] * x[1] + 2 * x[0] - 2, 200 * (x[1] - x[0] ** 2)])


# Гессиан функции Розенброка
def hessian_rosenbrock(x):
    return np.array([[2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]], [-400 * x[0], 200]])


# Неполиномиальная функция
def non_polynomial_function(x):
    return x[0] ** 2 + x[1] ** 2 + np.sin(x[0]) * np.cos(x[1])


# Градиент неполиномиальной функции
def grad_non_polynomial_function(x):
    df_dx = 2 * x[0] + np.cos(x[0]) * np.cos(x[1])
    df_dy = 2 * x[1] - np.sin(x[0]) * np.sin(x[1])
    return np.array([df_dx, df_dy])


# Гессиан неполиномиальной функции
def hessian_non_polynomial_function(x):
    d2f_dx2 = -np.sin(x[0]) * np.cos(x[1]) - np.sin(x[0]) * np.sin(x[1])
    d2f_dy2 = -np.cos(x[0]) * np.cos(x[1]) - np.cos(x[0]) * np.sin(x[1])
    d2f_dx_dy = -np.cos(x[0]) * np.sin(x[1]) + np.sin(x[0]) * np.cos(x[1])
    return np.array([[d2f_dx2, d2f_dx_dy], [d2f_dx_dy, d2f_dy2]])


# Начальная точка
# NOTE 1: x = [1, 1] - это точка минимума функции Розенброка, должно быть 0 итераций
# NOTE 2: при x0 = [0, 0] для неполиномиальной функции метод не сходится
x0 = [0, 0]

# Применяем метод оптимизации Ньютона с постоянным шагом к функции Розенброка
min_x, iters, taken_time, all_pointsRosC = newton_with_constant_learning_rate(rosenbrock, grad_rosenbrock, hessian_rosenbrock, x0)

# Выводим результат
print("Минимальная точка:", min_x)
print("Значение функции Розенброка в минимальной точке:", rosenbrock(min_x))
print("Количество итераций:", iters)
print("Время выполнения:", taken_time)
print()

x0 = [0.5, 0.5]
min_x, iters, taken_time, all_pointsNonPC = newton_with_constant_learning_rate(non_polynomial_function, grad_non_polynomial_function,
                                                              hessian_non_polynomial_function, x0)

# Выводим результат
print("Минимальная точка:", min_x)
print("Значение неполиномиальной функции в минимальной точке:", non_polynomial_function(min_x))
print("Количество итераций:", iters)
print("Время выполнения:", taken_time)
print()

x0 = [0, 0]
min_x, iters, taken_time, all_pointsRosS = newton_with_search_learning_rate(rosenbrock, grad_rosenbrock, hessian_rosenbrock, x0)

# Выводим результат
print("Минимальная точка:", min_x)
print("Значение функции Розенброка в минимальной точке:", rosenbrock(min_x))
print("Количество итераций:", iters)
print("Время выполнения:", taken_time)
print()

x0 = [0.5, 0.5]
min_x, iters, taken_time, all_pointsNonPS = newton_with_search_learning_rate(non_polynomial_function, grad_non_polynomial_function,
                                                            hessian_non_polynomial_function, x0)

# Выводим результат
print("Минимальная точка:", min_x)
print("Значение неполиномиальной функции в минимальной точке:", non_polynomial_function(min_x))
print("Количество итераций:", iters)
print("Время выполнения:", taken_time)
print()

# Графики
a, b = np.hsplit(all_pointsRosC, 2)
plot_graphs(rosenbrock, a, b)
a, b = np.hsplit(all_pointsNonPC, 2)
plot_graphs(non_polynomial_function, a, b)
a, b = np.hsplit(all_pointsRosS, 2)
plot_graphs(rosenbrock, a, b)
a, b = np.hsplit(all_pointsNonPS, 2)
plot_graphs(non_polynomial_function, a, b)


x0 = [0, 0]

min_x, iters, taken_time, _ = newton_cg(rosenbrock, grad_rosenbrock, x0)

print("Минимальная точка:", min_x)
print("Значение функции Розенброка в минимальной точке:", rosenbrock(min_x))
print("Количество итераций:", iters)
print("Время выполнения:", taken_time)
print()

x0 = [0.5, 0.5]

min_x, iters, taken_time, _ = quasinewton(non_polynomial_function, grad_non_polynomial_function, x0)

print("Минимальная точка:", min_x)
print("Значение неполиномиальной функции в минимальной точке:", non_polynomial_function(min_x))
print("Количество итераций:", iters)
print("Время выполнения:", taken_time)
print()

min_x, iters, taken_time, all_points = newton_with_wolfe_search(rosenbrock, grad_rosenbrock, hessian_rosenbrock, x0)

print("Минимальная точка:", min_x)
print("Значение функции Розенброка в минимальной точке:", rosenbrock(min_x))
print("Количество итераций:", iters)
print("Время выполнения:", taken_time)
