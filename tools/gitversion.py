#!/usr/bin/env python3

import subprocess


def get_git_version() -> str:
    """Get the current git version description."""
    try:
        version = (
            subprocess.check_output(['git', 'describe', '--long', '--tags', '--always'], stderr=subprocess.STDOUT)
            .decode('utf-8')
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        version = '0.0.0-0-g0000000'

    ver, n_commits, git_hash = version.rsplit('-', 2)
    if ver.startswith('v'):
        ver = ver[1:]
    if ver.count('.') == 1:
        ver = f'{ver}.0'
    if n_commits != '0':
        git_hash = git_hash.lstrip('g')
        ver = f'{ver}.dev{n_commits}+g{git_hash}'
    return ver


if __name__ == '__main__':
    print(get_git_version())
