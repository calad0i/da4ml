import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    'config',
    [['vivado', -1], ['quartus', -1], ['vitis', -1], ['vivado', 2], ['quartus', 2]],
)
def test_report(config, temp_directory):
    test_root = Path(os.environ.get('DA4ML_TEST_DIR', '/tmp/da4ml_test')) / 'test_data'
    backend, lc = config
    if backend == 'vitis':
        path = test_root / f'{backend}_example_model'
    else:
        path = test_root / f'{backend}_example_verilog_model_lc={lc}'

    subprocess.run(['da4ml', 'report', str(path)], capture_output=True, text=True, check=True)
    for ext in ['.html', '.json', '.csv', '.tsv', '.md']:
        out_path = Path(f'{temp_directory}/report{ext}')
        subprocess.run(['da4ml', 'report', str(path), '-o', out_path], capture_output=True, text=True, check=True)
        assert out_path.exists()
        out_path.unlink()
