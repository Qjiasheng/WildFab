#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// header define

torch::Tensor far_dist_check_cuda(
    const torch::Tensor surf_coords,  // (N_surf, 3)
    const torch::Tensor queries,      // (N_query, 3)
    const float dist_th               // distance threshold
);





