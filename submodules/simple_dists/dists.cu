#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include "utils.h"

// CUDA KERNEL
__global__ void far_mask_kernel(const float* __restrict__ surf, int Ns,
                                const float* __restrict__ qry,  int Nq,
                                float dist_th2,
                                bool* __restrict__ out_mask)
{
    const int q_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_id >= Nq) return;

    // current query point
    const float qx = qry[3 * q_id + 0];
    const float qy = qry[3 * q_id + 1];
    const float qz = qry[3 * q_id + 2];

    float min_d2 = FLT_MAX;

    // brute-force nearest neighbour search
    for (int s = 0; s < Ns; ++s)
    {
        const float sx = surf[3 * s + 0];
        const float sy = surf[3 * s + 1];
        const float sz = surf[3 * s + 2];

        const float dx = qx - sx;
        const float dy = qy - sy;
        const float dz = qz - sz;
        const float d2 = dx * dx + dy * dy + dz * dz;

        min_d2 = d2 < min_d2 ? d2 : min_d2;
        if (min_d2 < dist_th2)   // early exit: already “close”
            break;
    }
    out_mask[q_id] = (min_d2 > dist_th2);
}

torch::Tensor far_dist_check_cuda(const torch::Tensor surf_coords,
                                const torch::Tensor queries,
                                const float dist_th)
{

    const int Ns = surf_coords.size(0);
    const int Nq = queries.size(0);

    auto mask =torch::empty({Nq}, torch::dtype(torch::kBool).device(queries.device()));

    const int threads = 256;
    const int blocks  = (Nq + threads - 1) / threads;

    const float dist_th2 = dist_th * dist_th;

    far_mask_kernel<<<blocks, threads>>>(surf_coords.data_ptr<float>(), Ns,
                                         queries.data_ptr<float>(),     Nq,
                                         dist_th2,
                                         mask.data_ptr<bool>());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    return mask;
}