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
__device__ __forceinline__ scalar_t spike(scalar_t z, float vth = 0.3) {
  return z < vth ? 0.0 : 1.0;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_spike(scalar_t z) {
  const auto e = abs(z);
  const auto d_spike = e < 0.5 ? 1.0 : 0.0;
  return d_spike;
}

template <typename scalar_t>
__global__ void lif_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> output,
    const int input_size,
    const int batch_size,
    const float vth, 
    const float tau, 
    const int steps) {
  const int n = blockIdx.y;
  //const int n = threadIdx.x;
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  //const int s = blockIdx.x * blockDim.x + blockIdx.y;
  const int c = input.size(1);
  const int h = input.size(2);
  const int w = input.size(3);
  const int cindex = s / (h * w); 
  const int hindex = (s - cindex * c) / w;
  const int windex = (s - cindex * c - hindex * h);
  int i;

  if (s < input_size && n < batch_size) {
    auto voltage = input[n][cindex][hindex][windex][0];
    output[n][cindex][hindex][windex][0] = 100;//spike(voltage, vth);
    for (i=1; i<steps; i++) {
      //voltage = voltage * tau * (1 - output[n][cindex][hindex][windex][i-1]) + input[n][cindex][hindex][windex][i];
      
      //output[n][cindex][hindex][windex][i] = 
      //if (voltage > vth) {
      //  output[n][cindex][hindex][windex][i] = 0; 
      //  voltage = 0;
      //}
      //output[n][cindex][hindex][windex][i] = 100;//spike(voltage, vth);
    }
  }
}

template <typename scalar_t>
__global__ void lif_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> output,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> grad_input,
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
  const int windex = (s - cindex * c - hindex * h);
  int i, j;
      // dOn/dIm = dSpike * tau(1-O_n-1) * ... * tau(1-Om)
  if (s < input_size) {
    auto grad_lif = d_spike(input[n][cindex][hindex][windex][steps-1] - vth);
    
    for (i=steps-1; i>=0; i--) {
      grad_lif = d_spike(input[n][cindex][hindex][windex][i] - vth);
      grad_input[n][cindex][hindex][windex][i] = grad_output[n][cindex][hindex][windex][i] * grad_lif;
      for (j=i-1; j>=0; j--) {
        grad_input[n][cindex][hindex][windex][j] = grad_output[n][cindex][hindex][windex][i] * grad_lif;
        grad_lif = grad_lif * tau * (1 - output[n][cindex][hindex][windex][j]);
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

  const auto ndims = input.ndimension();
  //int steps = 0;
  //int input_size = 0;
  const auto steps = input.size(4);
  const int input_size = input.size(1) * input.size(2) * input.size(3); // C * H * W
  /*
  if (ndims == 3) { // fc
      steps = input.size(2);
      input_size = input.size(1);
  }
  else { // conv
      steps = input.size(4);
      input_size = input.size(1) * input.size(2) * input.size(3);
  }*/
  const auto batch_size = input.size(0);
  const int threads = 1024;
  const dim3 blocks((input_size + threads - 1) / threads, batch_size);
  //const dim3 threads(input.size(2), input.size(3));

  AT_DISPATCH_FLOATING_TYPES(output.type(), "lif_forward_cuda", ([&] {
    lif_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
        input_size, batch_size, vth, tau, steps);
  }));

  return output;
}

torch::Tensor lif_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor output,
    torch::Tensor input,
    const float vth, 
    const float tau) {
  auto grad_input = torch::zeros_like(input);

  const int batch_size = input.size(0);
  const int ndims = input.ndimension();
  //int input_size = 0;
  const auto steps = input.size(4);
  const auto input_size = input.size(1) * input.size(2) * input.size(3);
      /*
  if (ndims == 3) { // fc
    input_size = input.size(1);
  }
  else { // conv
    steps = input.size(4);
    input_size = input.size(1) * input.size(2) * input.size(3);
  }*/

  const int threads = 1024;
  const dim3 blocks((input_size + threads - 1) / threads, batch_size); // TODO

  AT_DISPATCH_FLOATING_TYPES(grad_input.type(), "lif_backward_cuda", ([&] {
    lif_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
        input.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
        grad_input.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
        input_size, vth, tau, steps);
  }));


  return grad_input;
}
