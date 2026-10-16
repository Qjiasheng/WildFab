#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

__global__ void neighbors_kernel(const float* p, const float* t, const float* n, const int64_t* cells, const int64_t* keys, const int64_t* order, const int64_t* dims, const int64_t* iso_ids, int64_t count, float radius, float alignment, float normal_alignment, int k, int64_t* ids, float* scores) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    // A search cell is one radius wide, so the 27 neighboring cells suffice.
    for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
        int64_t x = cells[3*i] + dx, y = cells[3*i+1] + dy, z = cells[3*i+2] + dz;
        if (x < 0 || y < 0 || z < 0 || x >= dims[0] || y >= dims[1] || z >= dims[2]) continue;
        const int64_t key = ((iso_ids[i] * dims[0] + x) * dims[1] + y) * dims[2] + z;
        int64_t lo = 0, hi = count;
        while (lo < hi) {
            int64_t mid = lo + (hi-lo)/2;
            if (keys[mid] < key) lo = mid+1; else hi = mid;
        }
        for (int64_t pos = lo; pos < count && keys[pos] == key; ++pos) {
            const int64_t j = order[pos];
            if (j == i) continue;
            float d[3], dist2 = 0, dot_i = 0, dot_j = 0, dot_n = 0;
            for (int a = 0; a < 3; ++a) {
                d[a] = p[3*j+a] - p[3*i+a]; dist2 += d[a]*d[a];
                dot_i += d[a]*t[3*i+a]; dot_j += d[a]*t[3*j+a]; dot_n += n[3*i+a]*n[3*j+a];
            }
            if (dist2 <= 0 || dist2 > radius*radius) continue;
            float distance = sqrtf(dist2), ai = fabsf(dot_i)/distance, aj = fabsf(dot_j)/distance;
            if (ai < alignment || aj < alignment || fabsf(dot_n) < normal_alignment) continue;
            float score = distance/radius + 2.f*(1.f-ai) + 2.f*(1.f-aj);
            const int64_t base = (2*i + (dot_i >= 0 ? 1 : 0))*k;
            for (int a = 0; a < k; ++a) {
                if (score < scores[base+a] || (score == scores[base+a] && j < ids[base+a])) {
                    for (int b = k-1; b > a; --b) {scores[base+b] = scores[base+b-1]; ids[base+b] = ids[base+b-1];}
                    scores[base+a] = score; ids[base+a] = j; break;
                }
            }
        }
    }
}

std::vector<torch::Tensor> score_neighbors_cuda(torch::Tensor points, torch::Tensor tangents, torch::Tensor normals, torch::Tensor cells, torch::Tensor keys, torch::Tensor order, torch::Tensor dims, torch::Tensor iso_ids, double radius, double alignment, double normal_alignment, int64_t k) {
    c10::cuda::CUDAGuard guard(points.device());
    const auto count = points.size(0);
    auto ids = torch::full({count, 2, k}, -1, points.options().dtype(torch::kInt64));
    auto scores = torch::full({count, 2, k}, INFINITY, points.options());
    if (count) {
        neighbors_kernel<<<(count+255)/256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(points.data_ptr<float>(), tangents.data_ptr<float>(), normals.data_ptr<float>(), cells.data_ptr<int64_t>(), keys.data_ptr<int64_t>(), order.data_ptr<int64_t>(), dims.data_ptr<int64_t>(), iso_ids.data_ptr<int64_t>(), count, radius, alignment, normal_alignment, k, ids.data_ptr<int64_t>(), scores.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {ids, scores};
}
