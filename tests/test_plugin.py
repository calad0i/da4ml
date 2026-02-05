import numpy as np
import pytest

from da4ml.converter import trace_model
from da4ml.converter.example import ExampleModel
from da4ml.trace import FixedVariableArrayInput, comb_trace


@pytest.mark.parametrize('inp_shape', [(4, 5), None])
@pytest.mark.parametrize('manual_inp_def', [False, True])
def test_plugin(inp_shape, manual_inp_def):
    model = ExampleModel(input_shape=inp_shape)

    if manual_inp_def:
        inputs = FixedVariableArrayInput((4, 5))
    else:
        inputs = None
        if inp_shape is None:
            pytest.xfail('If input cannot be auto inferred, manual input definition is required.')

    inp, out = trace_model(model, verbose=True, inputs=inputs)
    comb = comb_trace(inp, out)

    data_in = np.random.rand(1000, 4, 5).astype(np.float32) * 256 - 128
    r_np = np.array([model(x) for x in data_in], dtype=np.float32)
    r_comb = comb.predict(data_in).reshape(r_np.shape)
    assert np.all(r_np == r_comb)
