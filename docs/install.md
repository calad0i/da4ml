# Installation Guide

`da4ml` is available on PyPI as binary wheel and can be installed using pip.

```bash
pip install da4ml
```

## Requirements

### Binary Wheels

 - `python>=3.10` (3.13+ is recommended)
 - `numpy>=2.0`
 - `Linux on x86_64` or `MacOS on ARM64` platform

### Building from Source

 - `python>=3.10` (3.13+ is recommended)
 - `numpy`
 - A C++20 compliant compiler with OpenMP support
 - `meson-python>1.9.0`



```bash
git clone https://github.com/calad0i/da4ml.git
pip install ./da4ml
```

Alternatively, you can configure and build da4ml with meson directly:

```bash
meson setup build/cp31*
meson compile -C build/cp31*
```

```{warning}
If you are building an editable installation, the flag `[--no-build-isolation]` must be used. Since editable installation with meson is implemented by dynamic hooking of the `build_ext` command to recompile the C++ extension in-place, the build isolation will break the dynamic hooking and cause `ninja` to fail upon importing. You must also install `meson-python` manually before running `pip` in this case.
```

```{note}
Technically, you do not need `numpy>=2.0`, since that is due to the use of `nanobind`'s abi3 feature. However, since we do not see any good reason for using older numpy versions, we require `numpy>=2.0` for the binary wheels for simplicity. If you want to build from source with older numpy versions, you can modify the `meson.build` file to override the `limited_api` flag.
```
