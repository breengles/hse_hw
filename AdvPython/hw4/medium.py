import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from timeit import timeit


def get_f_x(args):
    f, x, logger = args
    logger.info(f"Par task @ x = {x} is started!")
    return f(x)


def integrate_par(f, a, b, engine, *, n_jobs=1, n_iter=1000, logger=None):
    step = (b - a) / n_iter
    args = [(f, a + i * step, logger) for i in range(n_iter)]
    with engine(max_workers=n_jobs) as t:
        res = list(t.map(get_f_x, args))

    return sum(res) * step


def integrate(f, a, b, *, n_jobs=1, n_iter=1000, logger=None):
    acc = 0
    step = (b - a) / n_iter
    for i in range(n_iter):
        logger.info(f"Seq task @ x = {i} is started!")
        acc += f(a + i * step) * step
    return acc


def get_logger(filename):
    logging.basicConfig(filename=filename, level=logging.INFO, format="%(asctime)s | %(message)s")
    return logging.getLogger(os.path.basename(__file__))


if __name__ == "__main__":
    n_iter = int(1e3)
    n_cpu = os.cpu_count()
    n_jobs_range = list(range(1, 2 * n_cpu + 1))

    names = ("seq", "threading", "processing")

    time_spent = {"seq": None, "threading": [], "processing": []}

    engines = (None, ThreadPoolExecutor, ProcessPoolExecutor)

    for name, engine in zip(names, engines):
        logger = get_logger("artifacts/log.txt")

        # print(integrate_par(math.cos, 0, math.pi / 2, engine, n_jobs=8, n_iter=n_iter, logger=logger))
        if name == "seq":
            time = timeit(lambda: integrate(math.cos, 0, math.pi / 2, n_iter=n_iter, logger=logger), number=1)
            time_spent[name] = time
        else:
            for n_jobs in n_jobs_range:
                time = timeit(
                    lambda: integrate_par(
                        math.cos, 0, math.pi / 2, engine, n_jobs=n_jobs, n_iter=n_iter, logger=logger
                    ),
                    number=1,
                )
                time_spent[name].append(time)

    with open("artifacts/time_spent.csv", "w+") as art:
        art.write("n_jobs,seq,threading,processing\n")

        for idx, n_jobs in enumerate(n_jobs_range):
            art.write(f"{n_jobs},{time_spent['seq']},{time_spent['threading'][idx]},{time_spent['processing'][idx]}\n")
