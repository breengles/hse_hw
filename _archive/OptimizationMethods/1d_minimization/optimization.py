from math import sin, cos, exp, log, pi, inf, copysign, isclose
from time import time
import numpy as np

phi = (5 ** 0.5 - 1) / 2


def f1(x):
    return x ** 2 / 2, x


def f2(x):
    return sin(x) + sin(10 / 3 * x), cos(x) + 10 / 3 * cos(10 / 3 * x)


def f3(x):
    val = 0
    dval = 0
    for k in range(1, 6):
        val -= k * sin((k + 1) * x + k)
        dval -= k * (k + 1) * cos((k + 1) * x + k)
    return val, dval


def f4(x):
    return -(16 * x ** 2 - 24 * x + 5) * exp(-x), (16 * x ** 2 - 56 * x + 29) * exp(-x)


def f5(x):
    return -(1.4 - 3 * x) * sin(18 * x), 3 * (18 * (x - 0.466667) * cos(18 * x) + sin(18 * x))


def f6(x):
    return -(x + sin(x)) * exp(-x ** 2), exp(-x ** 2) * (2 * x ** 2 + 2 * x * sin(x) - cos(x) - 1)


def f7(x):
    return sin(x) + sin(10 / 3 * x) + log(x) - 0.84 * x + 3, 1 / x + cos(x) + 10 / 3 * cos(10 / 3 * x) - 0.84


def f8(x):
    val = 0
    dval = 0
    for k in range(1, 7):
        val -= k * sin((k + 1) * x + k)
        dval -= k * (k + 1) * cos((k + 1) * x + k)
    return val, dval


def f9(x):
    return sin(x) + sin(2 / 3 * x), cos(x) + 2 / 3 * cos(2 / 3 * x)


def f10(x):
    return -x * sin(x), -sin(x) - x * cos(x)


def f11(x):
    return 2 * cos(x) + cos(2 * x), -2 * (sin(x) + sin(2 * x))


def f12(x):
    return sin(x) ** 3 + cos(x) ** 3, 3 * sin(x) * cos(x) * (sin(x) - cos(x))


def f13(x):
    return -x**(2/3) - (1 - x**2)**(1/3), 2/3*x/(1-x**2)**(2/3) - 2/(3*x**(1/3))


def f18(x):
    if x <= 3:
        return (x - 2) ** 2, 2 * (x - 2)
    else:
        return 2 * log(x - 2) + 1, 2 / (x - 2)


# Требуется реализовать метод: который будет находить минимум функции на отрезке [a, b]
def optimize(f, a: float, b: float, eps: float = 1e-8, max_iter=1000, x0=None):
    """
    Brent's method + 1st-order derivative
    WARNING: Если специфицировать x0, то возвращаться будет список 
    с элементами из кортежей (x, f_x, num_iter, abs(x - x0), twall)
    Без x0 будет возвращаться np.array([x]), где x --- лучшая точка.
    """
    tstart = time()

    if a > b:
        a, b = b, a

    # for data retrieving
    if x0 is not None:
        data = []

    # len of current previous and current intervals
    current_d = d = b - a

    # x -- best, w -- 2nd best, v -- 3rd best
    x = w = v = 0.5 * (a + b)  # init values
    val = f(x)  # init oracle call
    f_x = f_w = f_v = val[0]
    df_x = df_w = df_v = val[1]

    for num_iter in range(1, max_iter + 1):
        xm = 0.5 * (a + b)
        tol1 = eps * (abs(x) + 1e-1)
        tol2 = 2 * tol1
        # some tricky condition (see e.g. OptMeth ML lecture 2016 MSU (quasi-empirical)
        if abs(x - xm) + 0.5 * (b - a) <= tol2:
            break

        if abs(current_d) > tol1:
            # init out-of-bound d1 and d2
            d1 = d2 = 2 * (b - a)
            # try to fit parabola via 1st-order derivative
            if not isclose(x, w) and not isclose(df_w, df_x):
                d1 = (w - x) * df_x / (df_x - df_w)
            if not isclose(x, v) and not isclose(df_v, df_x):
                d2 = (v - x) * df_x / (df_x - df_v)

            u1 = x + d1
            u2 = x + d2

            # check if obtained points with parabola fitting are actually good enough
            is_u1_ok = (a - u1) * (u1 - b) > 0 >= df_x * d1
            is_u2_ok = (a - u2) * (u2 - b) > 0 >= df_x * d2

            prev_d, current_d = current_d, d

            if is_u1_ok or is_u2_ok:
                if is_u1_ok and is_u2_ok:
                    # take the smallest interval
                    d = d1 if abs(d1) < abs(d2) else d2
                elif is_u1_ok:
                    d = d1
                else:
                    d = d2
                if abs(d) <= abs(0.5 * prev_d):
                    u = x + d
                    if u - a < tol2 or b - u < tol2:
                        d = copysign(tol1, xm - x)
                else:
                    current_d = a - x if df_x >= 0.0 else b - x
                    d = 0.5 * current_d
            else:
                current_d = a - x if df_x >= 0.0 else b - x
                d = 0.5 * current_d
        else:
            current_d = a - x if df_x >= 0 else b - x
            d = 0.5 * current_d

        if abs(d) >= tol1:
            u = x + d
            f_u, df_u = f(u)  # oracle call <-|
        else: #                               | this will be actually either 1st or 2nd oracle call
            u = x + copysign(tol1, d)  #      | i.e. only 1 call is possible per iteration
            f_u, df_u = f(u)  # oracle call <-|
            if f_u > f_x:
                break

        if f_u <= f_x:
            if u >= x:
                a = x
            else:
                b = x
            v = w
            f_v = f_w
            df_v = df_w
            w = x
            f_w = f_x
            df_w = df_x
            x = u
            f_x = f_u
            df_x = df_u
        else:
            if u < x:
                a = u
            else:
                b = u
            if f_u <= f_w or isclose(w, x):
                v = w
                f_v = f_w
                df_v = df_w
                w = u
                f_w = f_u
                df_w = df_u
            elif f_u < f_v or isclose(v, x) or isclose(v, w):
                v = u
                f_v = f_u
                df_v = df_u
        twall = time() - tstart
        if x0 is not None:
            t = (x, f_x, num_iter, abs(x - x0), twall)
            data.append(t)

    if x0 is not None:
        return data

    print("Brent:", f'{f.__name__}: x = {x}, f_x = {f_x} with {num_iter} iteration(s)')
    # return x, f_x, num_iter, twall
    return np.array([x])


def golden_section(f, a: float, b: float, eps: float = 1e-8, max_iter=1000, x0=None):
    """
    WARNING: Если специфицировать x0, то возвращаться будет список 
    с элементами из кортежей (x, f_x, num_iter, abs(x - x0), twall)
    Без x0 будет возвращаться np.array([x]), где x --- лучшая точка.
    """
    tstart = time()
    if x0 is not None:
        data = []

    d = b - a
    d1 = d * phi
    x = l = b - d1
    r = a + d1
    f_l, df_l = f(l)
    f_r, df_r = f(r)
    for num_iter in range(1, max_iter + 1):
        d1 *= phi
        if f_l >= f_r:
            a = l
            l, r = r, a + d1
            x = r
            f_x = f_r
            f_l, df_l = f_r, df_r
            f_r, df_r = f(r)
        else:
            b = r
            l, r = b - d1, l
            x = l
            f_x = f_l
            f_r, df_r = f_l, df_l
            f_l, df_l = f(l)
        if d1 <= eps:
            if f_l <= f_r:
                x = l
                f_x = f_l
                break
            else:
                x = r
                f_x = f_r
                break
        twall = time() - tstart
        if x0 is not None:
            t = (x, f_x, num_iter, abs(x - x0), twall)
            data.append(t)

    if x0 is not None:
        return data

    return np.array([x])
    

# Задание состоит из 2-х частей— реализовать любой алгоритм оптимизации по выбору
# Провести анализ работы алгоритма на нескольких функция, построить графики сходимости вида:
# кол-во итераций vs log(точность); время работы vs log(точность)
# Изучить, как метод будет работать на неунимодальных функций и привести примеры, подтверждающие поведение
# (например, что будет сходится в ближайший локальный минимум)

# Критерий оценки:
# 4-5 баллов  — решение работает и дает правильный ответ,
# код реализации не вызывает вопрос + ipynb отчет с исследованием работы метода

# Оценка по дальнейшим результатам: будет 4-5 тестовых функций.
# На каждой будет для всех сданных решений строится распределение времени работы
# Далее по квантилям распределения: 10: 95%, 9: 85%, 8: 75%, 7: 50%
#   — по каждом заданию независимо, далее среднее по всем
# Дополнительно требование на 8+ баллов: минимальное требование обогнать бейзлайн-решение
# (скрыт от вас, простая наивная реализация одного из методов с лекции)


if __name__ == "__main__":
    # optimize(f1, -5, 5)
    # optimize(f2, 2.7, 7.5)
    # optimize(f3, -10, -5)
    # optimize(f4, 1.9, 3.9)
    # optimize(f5, 0, 1.2)
    # optimize(f6, -10, 10)
    # optimize(f7, 2.7, 7.5)
    # optimize(f8, -10, 10)
    # optimize(f9, 15, 20.4)
    # optimize(f10, 0, 10)
    # optimize(f11, -pi / 2, 2 * pi)
    optimize(f13, 0.001, 0.99)
