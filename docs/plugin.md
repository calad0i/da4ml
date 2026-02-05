# Conversion Plugin Documentation

To define a static computation graph, da4ml always use the numpy-style symbolic tracing internally. However, for user convenience, da4ml also provides a plugin system that allows users to convert models defined in other QAT frameworks into da4ml's static computation graph format. For any frontend to be supported, a conversion plugin needs to be implemented, either as a separate package or inside the frontend framework itself. da4ml itself only provides a minimal example plugin for testing and demonstration purpose.

## Plugin Interface

Any conversion plugin is defined in two parts: a model tracer class and an entry point defined in the wheel package metadata:
- **Model Tracer Class**: This class is responsible for tracing the model, or any generic dataflow definition, into numpy-like operations supported by da4ml. The tracer class needs to inherit from the abstract base class `da4ml.converter.plugin:DAISTracerPluginBase`, and implement the two abstract methods, `apply_model` and `get_input_shapes`.
- **Entry Point**: One needs to declare an entry point in the wheel package metadata under the group name `entry-points."dais_tracer.plugins".${base_framework_name}=${module_path}:${tracer_class_name}`. Here, `${base_framework_name}` is the name of the frontend framework (e.g., `keras` for Keras), and `${module_path}:${tracer_class_name}` is the module path and class name of the tracer class defined above.

When the two parts are defined, upon calling the `da4ml.converter.trace_model` function, one has may specify the `framework` argument to indicate which plugin to use for tracing the model. If the `framework` argument is not provided, da4ml will treat the base module in which the `model` object is defined (`type(model).__module__.split('.', 1)[0]`) as the framework name.

## Example Plugin

An example plugin class is provided in the `da4ml.converter.example` module for demonstration purpose. The entry point is also defined in the `pyproject.toml` file under the group name `entry-points."dais_tracer.plugins".da4ml = "da4ml.converter.example:ExampleDAISTracer"`. This plugin is used in the unit tests of the plugin system itself (`tests/test_plugin.py`), and one may refer to that test as an example.
