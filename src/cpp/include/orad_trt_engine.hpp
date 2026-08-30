// ORAD TensorRT inference engine (Phase 5 deployment framework).
//
// Generic ONNX -> TensorRT engine wrapper used to run the two exported graphs
// (bev_fusion.onnx, policy.onnx) on-device with FP16/INT8 and async CUDA
// streams. The perception engine produces the BEV feature tensor consumed by
// the policy engine; outputs feed the numpy safety filter (mirrored in C++ by
// src/safety_filter_node) before the Ackermann command is published.
//
// Build (CMake snippet):
//   find_package(CUDA REQUIRED)
//   find_package(CUDAToolkit REQUIRED)
//   find_package(TensorRT REQUIRED)         # /usr/include/x86_64-linux-gnu
//   add_library(orad_trt_engine src/orad_trt_engine.cpp)
//   target_link_libraries(orad_trt_engine PRIVATE nvinfer nvonnxparser CUDA::cudart)
#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace orad {

/// Minimal TensorRT logger (severity threshold configurable at construction).
class TrtLogger : public nvinfer1::ILogger {
 public:
  explicit TrtLogger(nvinfer1::ILogger::Severity lvl =
                         nvinfer1::ILogger::kWARNING)
      : level_(lvl) {}
  void log(nvinfer1::ILogger::Severity severity,
           const char* msg) noexcept override;

 private:
  nvinfer1::ILogger::Severity level_;
};

/// Generic ONNX -> TensorRT engine runner (perception / policy).
class TrtEngine {
 public:
  enum class Precision { kFP32, kFP16, kINT8 };

  explicit TrtEngine(Precision prec = Precision::kFP16, int device = 0);
  ~TrtEngine();

  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  /// Build an optimized engine from an ONNX model. For dynamic batch a single
  /// optimization profile [min=1, opt=8, max=32] is added; pass per-binding
  /// min/opt/max via \p profiles if finer control is needed.
  bool BuildFromOnnx(const std::string& onnx_path,
                     const std::string& cache_path = "");

  /// Serialize the current engine to disk (re-use across runs to skip rebuild).
  bool SaveEngine(const std::string& engine_path) const;
  /// Load a previously serialized engine (no ONNX parser needed at runtime).
  bool LoadEngine(const std::string& engine_path);

  /// Select optimization profile + set the active dynamic batch size.
  void SetProfile(int profile_index, int batch);

  /// Run one inference: \p inputs / \p outputs are device pointers in binding
  /// order (see NumBindings/BindingName). Pass a CUDA stream for async overlap
  /// with sensor I/O; nullptr runs synchronously.
  bool Infer(const std::vector<const void*>& inputs,
             const std::vector<void*>& outputs,
             cudaStream_t stream = nullptr);

  // Binding introspection (after Build/Load).
  int NumBindings() const;
  std::string BindingName(int i) const;
  bool IsInputBinding(int i) const;
  std::vector<std::int64_t> BindingShape(int i) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Convenience: chain perception + policy engines for one control tick.
/// `images`/`points`/`imu`/`attitude` are device pointers; `trajectory` is the
/// (batch, N, 4) policy output written to the provided device buffer.
struct PolicyPipeline {
  TrtEngine bev;
  TrtEngine policy;
  bool Step(const void* images, const void* points, const void* imu,
            const void* attitude, void* trajectory,
            cudaStream_t stream = nullptr);
};

}  // namespace orad
