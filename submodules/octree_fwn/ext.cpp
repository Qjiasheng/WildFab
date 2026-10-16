#include <torch/extension.h>
#include "includes/utils.h" 



std::tuple<std::vector<IntVec>, torch::Tensor, torch::Tensor>
build_octree(const torch::Tensor& points)
{
    // CHECK_INPUT(points);
    // leaf cell cloud number could affect performance
    int max_points_per_leaf = 40;
    int max_depth = 16;
    
    TORCH_CHECK(max_depth >= 0 && max_depth <= (WN_STACK_SIZE - 1) / 7,
                "max_depth exceeds the WN traversal stack capacity");
    return build_octree_cu(points, max_points_per_leaf, max_depth);
}

std::vector<torch::Tensor> initialize_octree(
  const torch::Tensor& points,
  const torch::Tensor& normals,
  const torch::Tensor& areas,
  const std::vector<IntVec>& point_indices,
  const torch::Tensor& children)
{
    return initialize_octree_cu(points, normals, areas, point_indices, children);
}



std::vector<torch::Tensor> fast_wn_clouds(

    const torch::Tensor& queries,          // (B,M,3)  float32/64
    const torch::Tensor& curr_ts,          // (B)      int32 / int64

    /* raw clouds ------------------------------------------------------- */
    const torch::Tensor& points,           // (P,3)
    const torch::Tensor& normals,          // (P,3)
    const torch::Tensor& areas,            // (P)

    /* leaf index data ----------------------------------------------- */
    const torch::Tensor& leaf_offsets,     // (N_nodes+1)
    const torch::Tensor& leaf_indices,     // ( #leaf points )

    /* octree aggregates --------------------------------------------- */
    const torch::Tensor& children,         // (N,8)  int64
    const torch::Tensor& cell_centres,     // (N,3)
    const torch::Tensor& cell_radii,       // (N)
    const torch::Tensor& cell_normals,     // (N,3)   sum a·n 
    const torch::Tensor& cell_time_iv,     // (N,2)   int32
    const float inv_delta
)
{
    CHECK_INPUT(queries);
    CHECK_INPUT(curr_ts);
    CHECK_INPUT(points);
    CHECK_INPUT(normals);
    CHECK_INPUT(areas);
    CHECK_INPUT(leaf_offsets);
    CHECK_INPUT(leaf_indices);
    CHECK_INPUT(children);
    CHECK_INPUT(cell_centres);
    CHECK_INPUT(cell_radii);
    CHECK_INPUT(cell_normals);
    CHECK_INPUT(cell_time_iv);

    float beta = 0.5f;
    return fast_wn_clouds_cu(queries, curr_ts, points, normals, areas, leaf_offsets, leaf_indices,
                             children, cell_centres, cell_radii, cell_normals, cell_time_iv, beta, inv_delta);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_octree", &build_octree, "build octree");
    m.def("initialize_octree", &initialize_octree, "initialize octree, we are leaf to root, they are root to leaf");
    m.def("fast_wn_clouds", &fast_wn_clouds, "fast winding number with point clouds");
}