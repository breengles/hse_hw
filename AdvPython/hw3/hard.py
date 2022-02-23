from matrix import HashedMatrix
import numpy as np


if __name__ == "__main__":
    eq_hashed = False
    not_eq_mat = False
    eq_mat = False
    not_eq_matmul = False

    while not (eq_hashed and not_eq_mat and eq_mat and not_eq_matmul):
        a_numpy = np.random.randint(0, 10, (2, 2))
        b_numpy = np.random.randint(0, 10, (2, 2))
        c_numpy = np.random.randint(0, 10, (2, 2))
        d_numpy = b_numpy.copy()

        a_hashed = HashedMatrix(a_numpy.tolist())
        b_hashed = HashedMatrix(b_numpy.tolist())
        c_hashed = HashedMatrix(c_numpy.tolist())
        d_hashed = HashedMatrix(d_numpy.tolist())

        ab = a_hashed @ b_hashed
        cd = c_hashed @ d_hashed

        eq_hashed = hash(ab) == hash(cd)
        not_eq_mat = a_hashed != c_hashed
        eq_mat = b_hashed == d_hashed
        not_eq_matmul = (a_hashed @ b_hashed) != (c_hashed @ d_hashed)

    with open("artifacts/hard/A.txt", "w+") as art:
        art.write(str(a_hashed))
    with open("artifacts/hard/B.txt", "w+") as art:
        art.write(str(b_hashed))
    with open("artifacts/hard/C.txt", "w+") as art:
        art.write(str(c_hashed))
    with open("artifacts/hard/D.txt", "w+") as art:
        art.write(str(d_hashed))

    print(eq_hashed and not_eq_mat and eq_mat and not_eq_matmul)

    with open("artifacts/hard/AB.txt", "w+") as art:
        art.write(str(ab))

    with open("artifacts/hard/CD.txt", "w+") as art:
        art.write(str(cd))

    with open("artifacts/hard/hash.txt", "w+") as art:
        art.write(str(hash(ab)) + "\n")
        art.write(str(hash(cd)))
