#include <torch/extension.h>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>   // for glm::abs(...)
#include <array>
#include <numeric>
#include <stack>
#include <tuple>
#include <vector>

// DFS with at most eight children needs 1 + 7 * max_depth entries.
#ifndef WN_STACK_SIZE
#define WN_STACK_SIZE 128
#endif

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


using IntVec  = std::vector<int>;
using Int8    = std::array<int, 8>;        // -1 = no child

std::tuple<std::vector<IntVec>, torch::Tensor>
build_octree_cu(const torch::Tensor& points,
                const int&  max_points_per_leaf,
                const int&  max_depth);


std::vector<torch::Tensor> initialize_octree_cu(
  const torch::Tensor& points,
  const torch::Tensor& normals,
  const torch::Tensor& areas,
  const std::vector<IntVec>& point_indices,
  const torch::Tensor& children);


std::vector<torch::Tensor> fast_wn_clouds_cu(

    const torch::Tensor& queries,          // (M,3)  float32/64

    const torch::Tensor& points,           // (P,3)
    const torch::Tensor& normals,          // (P,3)
    const torch::Tensor& areas,            // (P)

    const torch::Tensor& leaf_offsets,     // (N_nodes+1)
    const torch::Tensor& leaf_indices,     // ( #leaf points )

    const torch::Tensor& children,         // (N,8)  int64
    const torch::Tensor& cell_centres,     // (N,3)
    const torch::Tensor& cell_radii,       // (N)
    const torch::Tensor& cell_normals,     // (N,3)   Σ a·n
    const float beta,
    const float inv_delta
);