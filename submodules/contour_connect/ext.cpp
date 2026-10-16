#include <torch/extension.h>
#include <cmath>

std::vector<torch::Tensor> score_neighbors_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, double, double, int64_t);

std::vector<torch::Tensor> score_neighbors(torch::Tensor points, torch::Tensor tangents, torch::Tensor normals, torch::Tensor cells, torch::Tensor keys, torch::Tensor order, torch::Tensor dims, torch::Tensor iso_ids, double radius, double min_alignment, double min_normal_alignment, int64_t k) {
    for (const auto& t : {points, tangents, normals, cells, keys, order, dims, iso_ids}) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "inputs must be contiguous CUDA tensors");
        TORCH_CHECK(t.device() == points.device(), "inputs must share a device");
    }
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must be Nx3");
    TORCH_CHECK(tangents.sizes() == points.sizes() && normals.sizes() == points.sizes() && cells.sizes() == points.sizes(), "point attributes must be Nx3");
    TORCH_CHECK(points.scalar_type() == torch::kFloat32 && tangents.scalar_type() == torch::kFloat32 && normals.scalar_type() == torch::kFloat32, "geometry must be float32");
    TORCH_CHECK(cells.scalar_type() == torch::kInt64 && keys.scalar_type() == torch::kInt64 && order.scalar_type() == torch::kInt64 && dims.scalar_type() == torch::kInt64, "grid inputs must be int64");
    TORCH_CHECK(keys.dim() == 1 && order.dim() == 1 && keys.numel() == points.size(0) && order.numel() == points.size(0), "invalid sorted grid");
    TORCH_CHECK(dims.dim() == 1 && dims.numel() == 3, "dims must have 3 entries");
    TORCH_CHECK(iso_ids.scalar_type() == torch::kInt64 && iso_ids.dim() == 1 && iso_ids.numel() == points.size(0), "iso_ids must be int64 N-vector");
    TORCH_CHECK(std::isfinite(radius) && radius > 0 && min_alignment >= 0 && min_alignment <= 1 && min_normal_alignment >= 0 && min_normal_alignment <= 1 && k > 0 && k <= 16, "invalid search parameters");
    return score_neighbors_cuda(points, tangents, normals, cells, keys, order, dims, iso_ids, radius, min_alignment, min_normal_alignment, k);
}
std::vector<torch::Tensor> trace_graph_batch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trace_graph_batch", &trace_graph_batch, "Native parallel graph resolution and tracing", pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("score_neighbors", &score_neighbors, "Grid neighbor candidates, K per tangent side (CUDA)");
}
