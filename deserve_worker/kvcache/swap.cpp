#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include <c10/cuda/CUDAStream.h>


using CUDAStream = at::cuda::CUDAStream;

void swap(torch::Tensor &d, torch::Tensor &h0, torch::Tensor &h1, torch::Stream &s0, torch::Stream &s1, int interval) {
  if (interval <= 0) {
    return;
  }
  if (d.numel() != h0.numel() || d.numel() != h1.numel()) {
    throw std::runtime_error("Input and output tensors must have the same number of elements");
  }
  if (d.numel() % interval != 0 || h0.numel() % interval != 0 || h1.numel() % interval != 0) {
    throw std::runtime_error("Input and output tensors must be divisible by interval");
  }
  if (d.scalar_type() != h0.scalar_type() || d.scalar_type() != h1.scalar_type()) {
    throw std::runtime_error("Input and output tensors must have the same scalar type");
  }

  auto d_data = d.data_ptr<at::Half>();
  auto h0_data = h0.data_ptr<at::Half>();
  auto h1_data = h1.data_ptr<at::Half>();
  auto cs0 = CUDAStream(CUDAStream::UNCHECKED, s0);
  auto cs1 = CUDAStream(CUDAStream::UNCHECKED, s1);

  int total = d.numel();
  if (total % 2 != 0) {
    throw std::runtime_error("Input tensor must have an even number of elements");
  }
  int step = interval * sizeof(d.scalar_type());
  int mid = total / 2;
  cudaError_t err;
  for (int i = 0; i < mid; i += interval) {
    err = cudaMemcpyAsync(h0_data + i, d_data + i, step, cudaMemcpyDeviceToHost, cs0);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaMemcpyAsync(d_data + i, h1_data + i, step, cudaMemcpyHostToDevice, cs0);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
  }

  for (int i = mid; i < total; i += interval) {
    err = cudaMemcpyAsync(h0_data + i, d_data + i, step, cudaMemcpyDeviceToHost, cs1);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaMemcpyAsync(d_data + i, h1_data + i, step, cudaMemcpyHostToDevice, cs1);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swap", &swap, "Swap between device and host");
}
