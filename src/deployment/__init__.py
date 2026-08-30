"""Deployment: ONNX export & TensorRT runtime.

Modules:

* ``onnx_export`` - trace/export perception (:class:`BEVFusion`) and policy
  (:class:`HybridPolicy` deploy path) to ONNX (opset 17, dynamic batch).

Planned:

* ``tensorrt_engine`` - Python wrapper around the C++ TensorRT runtime in
  ``src/cpp/include/orad_trt_engine.hpp`` (FP16/INT8, async CUDA streams).
"""
