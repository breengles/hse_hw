from oracle import Oracle, generate_dataset, make_oracle
import numpy as np
import scipy


def test_grad(oracle, w, eps=1e-5):
    grad = []
    f = oracle.value
    for i in range(oracle.m):
        t = np.zeros(oracle.m)
        t[i] = eps
        grad.append((f(w + t) - f(w - t)) / (2 * eps))
    grad = np.array(grad)
    print("Grad test passed: ", np.allclose(grad, oracle.grad(w)))


def test_hessian(oracle, w, eps=1e-5):
    hessian = []
    true_hessian = oracle.hessian(w)
    f = oracle.value
    for i in range(oracle.m):
        ti = np.zeros(oracle.m)
        ti[i] = eps
        for j in range(oracle.m):
            tj = np.zeros(oracle.m)
            tj[j] = eps
            hessian.append((f(w + ti + tj) - f(w + ti) - f(w + tj) + f(w)) / eps ** 2)
    hessian = np.array(hessian).reshape(-1, oracle.m)
    print("Hessian test passed: ", end="")
    print(np.allclose(hessian, true_hessian))


def test_hessian_v2(oracle, w, eps=1e-5):
    num_hessian = []
    true_hessian = oracle.hessian(w)
    for i in range(oracle.m):
        p = np.zeros(oracle.m)
        p[i] = eps
        num_hessian.append((oracle.grad(w + p) - oracle.grad(w - p)) / (2 * eps))
    num_hessian = np.array(num_hessian).reshape(-1, oracle.m)
    print("Hessian test v2 passed: ", end="")
    print(np.allclose(num_hessian, true_hessian))


if __name__ == "__main__":
    print("\na1a")
    a1a = make_oracle("a1a.libsvm")
    w = np.ones(a1a.m)
    test_grad(a1a, w)
    test_hessian(a1a, w)
    test_hessian_v2(a1a, w)

    print("\nbc")
    bc = make_oracle("breast-cancer.libsvm")
    w = np.ones(bc.m)
    test_grad(bc, w)
    test_hessian(bc, w)
    test_hessian_v2(bc, w)

    print("\nbc_scaled")
    bc_scaled = make_oracle("breast-cancer_scale.libsvm")
    w = np.ones(bc_scaled.m)
    test_grad(bc_scaled, w)
    test_hessian(bc_scaled, w)
    test_hessian_v2(bc_scaled, w)
