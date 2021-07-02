#include <torch/extension.h>
#include <vector>

torch::Tensor lif_forward( 
        torch::Tensor input,
        float vth,
        float tau,
        float steps) {
    auto output = torch::ge(input, vth).toType(input.scalar_type());
    return output;
}


torch::Tensor lif_backward( 
        torch::Tensor grad_output,
        torch::Tensor input) {
    auto hu = torch::le(input.abs(), 0.5).toType(grad_output.scalar_type());
    hu = hu / (2 * 0.5);
    auto grad_input = grad_output.clone();
    grad_input = torch::mul(grad_input, hu);
    
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &lif_forward, "LIF forward");
    m.def("backward", &lif_backward, "LIF backward");
}