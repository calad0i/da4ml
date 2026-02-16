import numpy as np
from da4ml.cmvm.api import solve as solve_py
from tqdm import trange

from da4ml._binary import solve as solve_cpp

if __name__ == '__main__':
    cc, pp = [], []
    for _ in trange(1000):
        n, m = np.random.randint(1, 32, size=2)
        b = np.random.randint(1, 16)
        kernel = (np.random.rand(n, m).astype(np.float32) * 2**b - 2 ** (b - 1)).round()
        pipe_c = solve_cpp(kernel)
        pipe_p = solve_py(kernel)
        assert np.all(pipe_c.kernel == kernel) and np.all(pipe_p.kernel == kernel)
        cc.append(pipe_c.cost)
        pp.append(pipe_p.cost)

    pp, cc = np.array(pp), np.array(cc)
    diff, stddiv = np.mean(pp - cc), np.std(pp - cc) / np.sqrt(len(cc))
    print(diff, stddiv)
