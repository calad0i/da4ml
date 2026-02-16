import numpy as np
from tqdm import trange

from da4ml._binary import solve as solve_cpp

cc, pp = [], []
for _ in trange(5000):
    n, m = np.random.randint(1, 32, size=2)
    b = np.random.randint(1, 16)
    kernel = (np.random.rand(n, m).astype(np.float32) * 2**b - 2 ** (b - 1)).round()
    pipe_c = solve_cpp(kernel, method0='wmc', method1='wmc')
    assert np.all(pipe_c.kernel == kernel)
    cc.append(pipe_c.cost)
