import sys
from sysconfig import get_config_var

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext

openmp_compile_args = []
openmp_link_args = []
if sys.platform == 'win32':
    openmp_compile_args = ['/openmp']
else:
    openmp_compile_args = ['-fopenmp']
    openmp_link_args = ['-fopenmp']

cpp_extension = Extension(
    'da4ml._binary.libdais',
    sources=['src/da4ml/_binary/cpp/DAISInterpreter.cc'],
    extra_compile_args=list(openmp_compile_args),
    extra_link_args=list(openmp_link_args),
    language='c++',
)

# -----------------------------------------------------------


class BuildExtension(build_ext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_failed = False

    def get_ext_filename(self, fullname):
        """
        Overrides the default to build a file like _extension.so
        instead of _extension.cpython-3xx-xxxx.xx
        """

        ext_suffix = get_config_var('EXT_SUFFIX')
        lib_suffix = get_config_var('SHLIB_SUFFIX')

        if sys.platform == 'win32':
            lib_suffix = '.dll'

        filename: str = super().get_ext_filename(fullname)
        assert filename.endswith(ext_suffix)

        return filename[: -len(ext_suffix)] + lib_suffix

    def build_extensions(self):
        try:
            super().build_extensions()
            return
        except Exception:
            print('Extension compilation with OpenMP failed, retrying without OpenMP.')

        try:
            for ext in self.extensions:
                ext.extra_compile_args = [a for a in ext.extra_compile_args if a not in openmp_compile_args]
                ext.extra_link_args = [a for a in ext.extra_link_args if a not in openmp_link_args]
            super().build_extensions()
            return
        except Exception:
            print('Extension compilation failed without OpenMP as well.')

        self.build_failed = True

    def copy_extensions_to_source(self):
        if self.build_failed:
            print('Skipping extension copy due to build failure.')
            return
        super().copy_extensions_to_source()


setuptools.setup(
    ext_modules=[cpp_extension],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
