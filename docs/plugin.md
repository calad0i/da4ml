# Conversion Plugin

To define a static computation graph, da4ml always use the numpy-style symbolic tracing internally. However, for user convenience, da4ml also provides a plugin system that allows users to convert models defined in other QAT frameworks into da4ml's static computation graph format. For any frontend to be supported, a conversion plugin needs to be implemented, either as a separate package or inside the frontend framework itself. da4ml itself only provides a minimal example plugin for testing and demonstration purpose.

## Plugin Interface

Any conversion plugin is defined in two parts: a model tracer class and an entry point defined in the wheel package metadata:
- **Model Tracer Class**: This class is responsible for tracing the model, or any generic dataflow definition, into numpy-like operations supported by da4ml. The tracer class needs to inherit from the abstract base class `da4ml.converter.plugin:DAISTracerPluginBase`, and implement the two abstract methods, `apply_model` and `get_input_shapes`.
- **Entry Point**: One needs to declare an entry point in the wheel package metadata under the group name `entry-points."dais_tracer.plugins".${base_framework_name}=${module_path}:${tracer_class_name}`. Here, `${base_framework_name}` is the name of the frontend framework (e.g., `keras` for Keras), and `${module_path}:${tracer_class_name}` is the module path and class name of the tracer class defined above.

When the two parts are defined, upon calling the `da4ml.converter.trace_model` function, one may specify the `framework` argument to indicate which plugin to use for tracing the model. If the `framework` argument is not provided, da4ml will treat the base module in which the `model` object is defined (`type(model).__module__.split('.', 1)[0]`) as the framework name. For example, a Keras model (whose class is defined in `keras.*`) resolves to `'keras'`, and the plugin registered under `'keras'` is used automatically.

## Known Real-World Plugins

| Plugin    | Framework key | Package                                             | Description                                                                 |
| --------- | ------------- | --------------------------------------------------- | --------------------------------------------------------------------------- |
| HGQ2      | `keras`       | [HGQ2](https://github.com/calad0i/HGQ2)             | All HGQ2-provided, and most Keras3 provided Operation/Layers                |
| hls4ml    | `hls4ml`      | [da4ml_emu](https://github.com/calad0i/da4ml_emu)   | Converts hls4ml ModelGraph to DAIS, in progress and ok for experimental use |
| Torch/LGN | `torch`       | [Torch-LGN](https://github.com/calad0i/da4ml-torch) | Minimal work-in-progress plugin/skeleton                                    |

The [HGQ2](https://github.com/calad0i/HGQ2) package is the primary real-world plugin for da4ml. It registers a tracer under the `keras` key in the `dais_tracer.plugins` entry-point group, enabling `trace_model(keras_model)` to automatically convert HGQ2/Keras3 models. Installing HGQ2 is sufficient — no additional configuration is needed.

## Example Plugin

An example plugin class is provided in the `da4ml.converter.example` module for demonstration purpose. The entry point is also defined in the `pyproject.toml` file under the group name `entry-points."dais_tracer.plugins".da4ml = "da4ml.converter.example:ExampleDAISTracer"`. This plugin is used in the unit tests of the plugin system itself (`tests/test_plugin.py`), and one may refer to that test as an example.
