import numpy as np
from scipy.optimize import check_grad
import tasks


def test(task: tasks.Task, n: int = 5, verbose=False):
    x = np.random.uniform(size=(n, 1))

    if verbose:
        print("Value")
        print(task.f(x))

        print("True grad")
        print("norm = ", np.linalg.norm(task.df(x)))
        print(task.df(x))
        print("numgra")
        print("norm = ", np.linalg.norm(task.numgra(x)))
        print(task.numgra(x))

        print("True hessian")
        print("norm = ", np.linalg.norm(task.ddf(x)))
        print(task.ddf(x))
        print("Numhes")
        print("norm = ", np.linalg.norm(task.numhes(x)))
        print(task.numhes(x))

    grad_err = np.linalg.norm(task.df(x) - task.numgra(x)) / np.linalg.norm(task.df(x))
    hes_err = np.linalg.norm(task.ddf(x) - task.numhes(x)) / np.linalg.norm(task.ddf(x))
    print(f"{task.name:>7} | {grad_err:>.8e} | {hes_err:>.8e} |")


if __name__ == "__main__":
    print("   Task | ||g* - g_num|| | ||H* - H_num|| |")
    print("--------|----------------|----------------|")

    n = 100
    
    # test(tasks.Task43(n), n, verbose=False)
    
    tasks_ = [tasks.Task0(n), tasks.Task41(n), tasks.Task42(n), tasks.Task43(n)]

    for task in tasks_:
        test(task, n)

    test(tasks.Task61(), 2)

    for lam in range(-1, 4):
        task62 = tasks.Task62(lam)
        task62.name = f"{task62.name}:{lam}"
        test(task62, 2)

    task63 = tasks.Task42(n)
    task63.name = "6.3"
    test(task63, n)
