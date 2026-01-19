project('da4ml', 'cpp',
  version: run_command('python3', 'tools/gitversion.py', check: true).stdout().strip(),
  meson_version: '>=1.9.0',
  default_options: ['buildtype=release', 'cpp_std=c++20'],
  license: 'LGPL-3.0-or-later',
)

py = import('python').find_installation(pure: false)

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

dais_bin = py.extension_module(
  'dais_bin',
  sources: [
    'src/da4ml/_binary/dais/bindings.cc',
    'src/da4ml/_binary/dais/DAISInterpreter.cc',
  ],
  dependencies: [nanobind_dep, omp_dep],
  install: true,
  subdir: 'da4ml/_binary',
  cpp_args: ['-Wno-sign-compare', '-Wno-unused-variable'],
)

install_subdir('src/da4ml',
  install_dir: py.get_install_dir(),
  exclude_directories: ['_binary/dais', '__pycache__'],
  exclude_files: ['_version.py.in']
)

stubgen = nanobind_dep.get_variable('stubgen')
custom_target(
  output: 'dais_bin.pyi',
  depends: dais_bin,
  command: [py, stubgen, '-m', 'dais_bin', '-M', 'py.typed'],
  install:true,
  install_dir: py.get_install_dir() / 'da4ml' / '_binary',
)
