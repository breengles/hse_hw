from multiprocessing import Process
from threading import Thread
from timeit import timeit


def fib(n):
    prev, cur = 0, 1
    res = [prev]

    for _ in range(n - 1):
        prev, cur = cur, prev + cur
        res.append(prev)

    return res


def fib_par(engine=Thread, n=int(1e5), num_jobs=10):
    jobs = []
    for _ in range(num_jobs):
        jobs.append(engine(target=fib, args=(n,)))
        jobs[-1].start()

    for job in jobs:
        job.join()


def seq_time(n=int(1e5), k=10):
    return timeit(lambda: fib(n), number=k)


def threading_time(n=int(1e5), k=10):
    return timeit(lambda: fib_par(Thread, n), number=k)


def multiprocessing_time(n=int(1e5), k=10):
    return timeit(lambda: fib_par(Process, n), number=k)


if __name__ == "__main__":
    n = int(1e5)
    k = 10

    with open("artifacts/easy.txt", "w+") as art:
        art.write(f"n = {n}\n")
        art.write(f"sequential timeit with {k} runs: {seq_time(n, k)} s.\n")
        art.write(f"threading timeit with {k} runs: {threading_time(n, 1)} s.\n")
        art.write(f"multiprocessing timeit with {k} runs: {multiprocessing_time(n, 1)} s.\n")
