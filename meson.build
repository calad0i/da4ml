project('da4ml', 'cpp',
  version: run_command('python3', 'tools/gitversion.py').stdout().strip(),
  meson_version: '>=1.9.0',
  default_options: ['buildtype=release', 'cpp_std=c++20', 'cpp_args=-std=c++20'],
  license: 'LGPL-3.0-or-later',
)

py = import('python').find_installation(pure: false)

min_numpy_version = '1.26.4'
min_python_version = '3.10'

vcs_tag(
  command: ['python3', 'tools/gitversion.py'],
  input: 'src/da4ml/_version.py.in',
  output: '_version.py',
  install_dir: py.get_install_dir() / 'da4ml',
  replace_string: '@VCS_TAG@',
  install: true,
)

nanobind_dep = dependency('nanobind')
omp_dep = dependency('openmp')

py.extension_module(
  'dais_bin',
  sources: [
    'src/da4ml/_binary/cpp/bindings.cc',
    'src/da4ml/_binary/cpp/DAISInterpreter.cc',
  ],
  dependencies: [nanobind_dep, omp_dep],
  install: true,
  subdir: 'da4ml/_binary',
  cpp_args: ['-std=c++20'],
)

install_subdir('src/da4ml',
  install_dir: py.get_install_dir(),
  exclude_directories: ['_binary/cpp', '__pycache__'],
  exclude_files: ['_version.py.in']
)
