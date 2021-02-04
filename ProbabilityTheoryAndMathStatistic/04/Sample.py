import numpy as np


class Sample:
    def __init__(self, n, s, kind="uniform"):
        self.kind = kind.strip()
        if self.kind == "uniform":
            self.xs = np.random.uniform(size=(s, n))  # U[0,1)
        elif self.kind == "exp":
            self.xs = np.random.exponential(size=(s, n))  # Exp(1)
        else:
            raise NotImplementedError("Unknown type of distribution")

    def get_thetas(self, k: int) -> np.ndarray:
        if k < 1:
            raise ValueError(f"Incorrect k: passed {k} but should be > 0")

        if self.kind == "uniform":
            return np.power(np.mean(np.power(self.xs, k), axis=1) * (k + 1), 1 / k)
        elif self.kind == "exp":
            if k <= 170:
                return np.power(np.mean(np.power(self.xs, k), axis=1) / np.math.factorial(k), 1 / k)
            else:
                # Stirling
                return (
                    np.exp(1) / k 
                    * np.power(
                        np.mean(np.power(self.xs, k), axis=1) 
                        * np.power(np.sqrt(2 * np.pi * k), -0.5), 1 / k
                    )
                )

    def get_theta_rms(self, k) -> float:
        return np.sqrt(np.mean(np.power(self.get_thetas(k) - 1, 2)))

    def to_plot(self, k_max: int) -> list:
        data = []
        for i in range(1, k_max + 1):
            data.append(self.get_theta_rms(i))

        return data
