project('xtensor', 'cpp',
    version : '0.27.1',
    license : 'BSD-3-Clause',
    default_options : ['cpp_std=c++20'])

xtensor_deps = []
xtensor_args = []

# Core dependency
xtl_dep = dependency('xtl', version : '>=0.8.0')
xtensor_deps += xtl_dep

# Optional dependencies
nlohmann_json_dep = dependency('nlohmann_json', version : '>=3.1.1', required : false)
if nlohmann_json_dep.found()
  xtensor_deps += nlohmann_json_dep
endif

if get_option('use_xsimd')
  xsimd_dep = dependency('xsimd', version : '>=13.2.0')
  xtensor_deps += xsimd_dep
endif

if get_option('use_tbb')
  if get_option('use_openmp')
    error('XTENSOR_USE_TBB and XTENSOR_USE_OPENMP cannot both be active at once')
  endif
  tbb_dep = dependency('tbb')
  xtensor_deps += tbb_dep
endif

if get_option('use_openmp')
  openmp_dep = dependency('openmp')
  xtensor_deps += openmp_dep
endif

# Compiler definitions
if get_option('enable_assert') or get_option('check_dimension')
  xtensor_args += '-DXTENSOR_ENABLE_ASSERT'
endif

if get_option('check_dimension')
  xtensor_args += '-DXTENSOR_ENABLE_CHECK_DIMENSION'
endif

if get_option('force_temporary_memory')
  xtensor_args += '-DXTENSOR_FORCE_TEMPORARY_MEMORY_IN_ASSIGNMENTS'
endif

if get_option('default_column_major')
  xtensor_args += '-DXTENSOR_DEFAULT_LAYOUT=layout_type::column_major'
endif

if get_option('disable_exceptions')
  xtensor_args += '-DXTENSOR_DISABLE_EXCEPTIONS'
endif

# Interface library
xtensor_inc = include_directories('include')

xtensor_dep = declare_dependency(
  include_directories : xtensor_inc,
  dependencies : xtensor_deps,
  compile_args : xtensor_args
)
