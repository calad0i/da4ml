project('da4ml', 'cpp',
  version: run_command('python3', 'tools/gitversion.py', check: true).stdout().strip(),
  meson_version: '>=1.9.0',
  default_options: ['buildtype=release', 'cpp_std=c++20'],
  license: 'LGPL-3.0-only',
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
xtensor_dep = dependency('xtensor', default_options: ['use_xsimd=true', 'use_tbb=false', 'use_openmp=true'])

_limited_abi = py.language_version().version_compare('>=3.12') ? '3.12' : ''

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
  limited_api: _limited_abi,
)

cmvm_bin = py.extension_module(
  'cmvm_bin',
  sources: [
    'src/da4ml/_binary/cmvm/api.cc',
    'src/da4ml/_binary/cmvm/bindings.cc',
    'src/da4ml/_binary/cmvm/bit_decompose.cc',
    'src/da4ml/_binary/cmvm/cmvm_core.cc',
    'src/da4ml/_binary/cmvm/indexers.cc',
    'src/da4ml/_binary/cmvm/mat_decompose.cc',
    'src/da4ml/_binary/cmvm/state_opr.cc',
    'src/da4ml/_binary/dais/bindings.cc',
    'src/da4ml/_binary/dais/DAISInterpreter.cc',
  ],
  dependencies: [nanobind_dep, xtensor_dep, omp_dep],
  install: true,
  subdir: 'da4ml/_binary',
  cpp_args: ['-Wno-sign-compare', '-Wno-unused-variable'],
  limited_api: _limited_abi,
)

install_subdir('src/da4ml',
  install_dir: py.get_install_dir(),
  exclude_directories: [
    '_binary',
    '**/__pycache__',
    'codegen/hls/source/ap_types',
    'codegen/hls/source/ac_types'
  ],
  exclude_files: ['_version.py.in']
)

install_data('LICENSE', install_dir: py.get_install_dir() / 'da4ml')

install_data(
  'src/da4ml/_binary/__init__.py',
  install_dir: py.get_install_dir() / 'da4ml' / '_binary',
)

install_subdir('src/da4ml/codegen/hls/source/ac_types/include',
  install_dir: py.get_install_dir() / 'da4ml/codegen/hls/source/ac_types',
)

install_subdir('src/da4ml/codegen/hls/source/ap_types/include',
  install_dir: py.get_install_dir() / 'da4ml/codegen/hls/source/ap_types',
)

stubgen = nanobind_dep.get_variable('stubgen')

custom_target(
  output: 'dais_bin.pyi',
  depends: dais_bin,
  command: [py, stubgen, '-m', 'dais_bin', '-M', 'py.typed'],
  install: true,
  install_dir: py.get_install_dir() / 'da4ml' / '_binary',
)

custom_target(
  output: 'cmvm_bin.pyi',
  depends: cmvm_bin,
  command: [py, stubgen, '-m', 'cmvm_bin', '-M', 'py.typed'],
  install: true,
  install_dir: py.get_install_dir() / 'da4ml' / '_binary',
)
