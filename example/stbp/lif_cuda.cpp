#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor lif_cuda_forward(
        torch::Tensor input,
        const float vth,
        const float tau);

torch::Tensor lif_cuda_backward(
        torch::Tensor grad_output,
        torch::Tensor output,
        torch::Tensor input,
        const float vth, 
        const float tau);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor lif_forward(
        torch::Tensor input,
        const float vth,
        const float tau) {
  CHECK_INPUT(input);

  return lif_cuda_forward(input, vth, tau);
}

torch::Tensor lif_backward(
        torch::Tensor grad_output,
        torch::Tensor output,
        torch::Tensor input,
        const float vth, 
        const float tau) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(input);

  return lif_cuda_backward(grad_output, output, input, vth, tau);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lif_forward, "LIF forward (CUDA)");
  m.def("backward", &lif_backward, "LIF backward (CUDA)");
}
