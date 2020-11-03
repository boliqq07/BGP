from timeit import default_timer as timer

import numpy as np
import torch
from numba import vectorize

print(torch.cuda.is_available())
torch.cuda.get_device_name(0)


@vectorize(["float32(float32, float32)"], target='cuda')
def vectorAdd(a, b):
    s = 0
    for _ in range(1000):
        if a > 0:
            t = a * b
        else:
            t = a
        s = t + s
    return s


@vectorize(["float32(float32, float32)"], target='cpu')
def npAdd(a, b):
    s = 0
    for _ in range(1000):
        if a > 0:
            t = a * b
        else:
            t = a
        s = t + s
    return s


def main():
    A = np.random.random((20000, 2000), ) - 0.5
    A = A.astype(dtype=np.float32)
    B = np.ones((20000, 2000), dtype=np.float32)
    C = np.zeros((20000, 2000), dtype=np.float32)

    start = timer()
    C = vectorAdd(A, B)
    vectorAdd_time = timer() - start

    print("vectorAdd took %f seconds " % vectorAdd_time)

    start = timer()
    C = npAdd(A, B)
    numbaAdd_time = timer() - start

    print("numbaadd %f seconds " % numbaAdd_time)


if __name__ == '__main__':
    main()
