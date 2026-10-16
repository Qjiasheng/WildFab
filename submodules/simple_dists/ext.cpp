#include <torch/extension.h>
#include "utils.h" 

torch::Tensor far_dist_check(
    const torch::Tensor surf_coords,
    const torch::Tensor queries,
    const float dist_th
){
    CHECK_INPUT(surf_coords);
    CHECK_INPUT(queries);

    return far_dist_check_cuda(surf_coords, queries, dist_th);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("far_dist_check", &far_dist_check);
}