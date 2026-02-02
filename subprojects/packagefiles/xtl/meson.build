project(
  'xtl',
  'cpp',
  version: '0.8.0',
  meson_version: '>=0.49.0',
  license: 'BSD-3-Clause',
  default_options: ['warning_level=3'],
)

json_dep = dependency(
  'nlohmann_json',
  version: '>=3.1.1',
)

xtl_dep = declare_dependency(
  include_directories: include_directories(
    'include',
    is_system: true,
  ),
  dependencies: [json_dep],
)
