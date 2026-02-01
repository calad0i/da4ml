import os
from pathlib import Path

import pytest


@pytest.fixture(scope='function')
def temp_directory(request: pytest.FixtureRequest):
    root = Path(os.environ.get('DA4ML_TEST_DIR', '/tmp/da4ml_test'))
    root.mkdir(exist_ok=True)

    test_name = request.node.name
    cls_name = request.cls.__name__ if request.cls else None
    if cls_name is None:
        test_dir = root / test_name
    else:
        test_dir = root / f'{cls_name}.{test_name}'
    test_dir.mkdir(exist_ok=True)
    return str(test_dir)


def pytest_sessionfinish(session, exitstatus):
    """whole test run finishes."""
    root = Path(os.environ.get('DA4ML_TEST_DIR', '/tmp/da4ml_test'))
    # Purge empty directories
    for path in root.glob('*'):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
