#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmaxf(0.0, z) + fminf(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}

template <typename scalar_t>
__global__ void lif_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits, size_t> input,
    torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits, size_t> output,
    const int input_size,
    const float vth, 
    const float tau, 
    const int steps) {
  const int n = blockIdx.y;
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = input.size(1);
  const int h = input.size(2);
  const int w = input.size(3);
  const int cindex = s / (h * w); 
  const int hindex = (s - cindex * c) / w;
  const int windex = (s - cindex * c - h_index * h);
  int i;

  if (s < input_size) {
    const auto voltage = input[n][cindex][hindex][windex][0];
    if (voltage > vth) {
      output[n][cindex][hindex][windex][0] = 0;
      voltage = 0;
    }
    for (i=1; i<steps; i++) {
      voltage = voltage * tau;
      voltage = voltage + input[n][cindex][hindex][windex][i];
      if (voltage > vth) {
        output[n][cindex][hindex][windex][i] = 0; 
        voltage = 0;
      }
    }
  }
}

template <typename scalar_t>
__global__ void lif_cuda_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits, size_t> grad_output,
    const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits, size_t> input,
    torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits, size_t> grad_input,
    const int input_size);) {
  const int n = blockIdx.y;
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = input.size(1);
  const int h = input.size(2);
  const int w = input.size(3);
  const int cindex = s / (h * w); 
  const int hindex = (s - cindex * c) / w;
  const int windex = (s - cindex * c - h_index * h);
  int i;

  if (s < input_size) {
    const auto voltage = input[n][cindex][hindex][windex][0];
    if (voltage > vth) {
      output[n][cindex][hindex][windex][0] = 0;
      voltage = 0;
    }
    for (i=1; i<steps; i++) {
      voltage = voltage * tau;
      voltage = voltage + input[n][cindex][hindex][windex][i];
      if (voltage > vth) {
        output[n][cindex][hindex][windex][i] = 0; 
        voltage = 0;
      }
    }
  }



  }
}
} // namespace

torch::Tensor lif_cuda_forward(
    torch::Tensor input,
    const float vth,
    const float tau) {
  auto output = torch::zeros_like(input);

  const int ndims = input.ndimension();
  int steps = 0;
  int input_size = 0;
  if (ndims == 3) { // fc
      steps = input.size(2);
      input_size = input.size(1);
  }
  else { // conv
      steps = input.size(4);
      input_size = input.size(1) * input.size(2) * input.size(3);
  }
  const int batch_size = input.size(0);
  //一般一个block 1024个thread。看你怎么分block
  const int threads = 1024;
  const dim3 blocks((input_size + threads - 1) / threads, batch_size);
  //const dim3 threads(input.size(2), input.size(3));

  AT_DISPATCH_FLOATING_TYPES(input.type(), "lif_forward_cuda", ([&] {
    lif_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t, dims, torch::RestrictPtrTraits, size_t>(),
        output.packed_accessor<scalar_t, dims, torch::RestrictPtrTraits, size_t>(),
        input_size, vth, tau, steps);
  }));

  return output;
}

torch::Tensor lif_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input) {
  auto grad_input = torch::zeros_like(input);

  const int batch_size = input.size(0);
  const int ndims = input.ndimension();
  int input_size = 0;

  if (ndims == 3) { // fc
    input_size = input.size(1);
  }
  else { // conv
    input_size = input.size(1) * input.size(2) * input.size(3);
  }

  const int threads = 1024;
  const dim3 blocks((input_size + threads - 1) / threads, batch_size); // TODO

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "lif_forward_cuda", ([&] {
    lif_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.packed_accessor<scalar_t, dims, torch::RestrictPtrTraits, size_t>(),
        input.packed_accessor<scalar_t, dims, torch::RestrictPtrTraits, size_t>(),
        grad_input.packed_accessor<scalar_t, dims, torch::RestrictPtrTraits, size_t>(),
        input_size);
  }));


  return grad_input;
}
