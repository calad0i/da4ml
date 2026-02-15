import fcntl
import os
import sys
import tarfile
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
    # Skip on xdist worker nodes
    if hasattr(session.config, 'workerinput'):
        return
    root = Path(os.environ.get('DA4ML_TEST_DIR', '/tmp/da4ml_test'))
    # Purge empty directories
    if exitstatus == 0:
        os.system(f'rm -rf {root}')
        return
    for path in root.glob('*'):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    if not any(root.iterdir()):
        root.rmdir()


def pytest_collection_finish(session):
    if not any('test_report' in str(item) for item in session.items):
        return
    root = Path(os.environ.get('DA4ML_TEST_DIR', '/tmp/da4ml_test'))
    root.mkdir(exist_ok=True)

    lock = root / '.extract_lock'
    with open(lock, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if (root / 'test_data').exists():
                return
            kwargs = {'filter': 'data'} if sys.version_info >= (3, 12) else {}
            with tarfile.open(Path(__file__).parent / 'test_data.tar.xz', 'r:xz') as tar:
                tar.extractall(root, **kwargs)  # type: ignore
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
