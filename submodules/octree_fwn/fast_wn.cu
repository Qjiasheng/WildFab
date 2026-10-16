#include <torch/extension.h>
#include <vector>
#include <cmath>
#include "includes/utils.h"

#define ONE_OVER_PI4       0.079577471f
#define TWO_OVER_SQRT_PI   1.1283791671f
#define ONE_OVER_SQRT_PI   0.5641895835f

template <typename scalar_t>
__device__ inline scalar_t dot3(const scalar_t *a, const scalar_t *b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename scalar_t>
__device__ inline void sub3(const scalar_t *a, const scalar_t *b, scalar_t *out)
{
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

template <typename scalar_t>
__device__ inline scalar_t inv_r3(scalar_t r2)
{
    scalar_t r = sqrt(r2);
    return scalar_t(1) / (r2 * r + scalar_t(1e-20));
}

template <typename scalar_t>
__device__ inline void add_grad_contrib(
    const scalar_t dvec[3],
    const scalar_t avec[3],
    const scalar_t area_scale,
    const scalar_t dist2,
    const scalar_t s,
    const scalar_t dphi_dt,
    const scalar_t inv_delta,
    scalar_t &field_acc,
    scalar_t grad_acc[3]
)
{
    const scalar_t dist = sqrt(dist2);
    const scalar_t dn = area_scale * dot3<scalar_t>(dvec, avec);

    const scalar_t inv_r3_val = scalar_t(1) / (dist2 * dist + scalar_t(1e-20));
    const scalar_t poisson = dn * inv_r3_val;

    scalar_t ds_dv[3] = {scalar_t(0), scalar_t(0), scalar_t(0)};
    if (dphi_dt != scalar_t(0))
    {
        const scalar_t coeff = dphi_dt * inv_delta / (dist + scalar_t(1e-20));
        ds_dv[0] = coeff * dvec[0];
        ds_dv[1] = coeff * dvec[1];
        ds_dv[2] = coeff * dvec[2];
    }

    const scalar_t denom = dist2 * dist2 * dist + scalar_t(1e-20);

    scalar_t dp_dv[3];
    dp_dv[0] = (dist2 * area_scale * avec[0] - scalar_t(3) * dn * dvec[0]) / denom;
    dp_dv[1] = (dist2 * area_scale * avec[1] - scalar_t(3) * dn * dvec[1]) / denom;
    dp_dv[2] = (dist2 * area_scale * avec[2] - scalar_t(3) * dn * dvec[2]) / denom;

    field_acc += poisson * s;

    grad_acc[0] += ds_dv[0] * poisson + s * dp_dv[0];
    grad_acc[1] += ds_dv[1] * poisson + s * dp_dv[1];
    grad_acc[2] += ds_dv[2] * poisson + s * dp_dv[2];
}


// regularization Poisson kernel 
template <typename scalar_t>
__device__ inline void eval_window(
    const scalar_t dist2,
    const scalar_t inv_delta,
    scalar_t &s,
    scalar_t &dphi_dt
)
{
    const scalar_t dist = sqrt(dist2);
    const scalar_t t = dist * inv_delta;
    const scalar_t t2 = t * t;
    const scalar_t exp_term = exp(-t2);

    s = erf(t) - scalar_t(TWO_OVER_SQRT_PI) * t * exp_term;
    dphi_dt = scalar_t(4) * t2 * scalar_t(ONE_OVER_SQRT_PI) * exp_term;
}

template <typename scalar_t>
__global__ void fast_wn_kernel(

    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> queries,
    torch::PackedTensorAccessor32<int64_t,1,torch::RestrictPtrTraits> curr_ts,

    torch::PackedTensorAccessor32<int64_t,2,torch::RestrictPtrTraits> children,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> cell_c,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> cell_r,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> cell_an,
    torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> cell_tiv,

    torch::PackedTensorAccessor32<int64_t,1,torch::RestrictPtrTraits> leaf_offs,
    torch::PackedTensorAccessor32<int64_t,1,torch::RestrictPtrTraits> leaf_idx,

    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> normals,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> areas,

    const scalar_t beta,
    const scalar_t inv_delta,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> out_field,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out_grad
)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= queries.size(0) * queries.size(1)) return;

    const int b = t / queries.size(1);
    const int m = t % queries.size(1);

    const int64_t curT = curr_ts[b];

    scalar_t q[3] = {
        queries[b][m][0],
        queries[b][m][1],
        queries[b][m][2]
    };

    int stack[WN_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;

    scalar_t field_acc = 0;
    scalar_t grad_acc[3] = {
        scalar_t(0),
        scalar_t(0),
        scalar_t(0)
    };

    while (sp)
    {
        const int n = stack[--sp];

        const int tmin = cell_tiv[n][0];
        const int tmax = cell_tiv[n][1];

        if ((tmax < 0) && (tmin < 0))
            continue;

        if (tmin > curT)
            continue;

        const bool fully_inside = tmax <= curT;

        scalar_t dvec[3];
        sub3<scalar_t>(&cell_c[n][0], q, dvec);

        scalar_t dist2 = dot3<scalar_t>(dvec, dvec) + scalar_t(1e-8);
        const bool far_enough = (cell_r[n] * cell_r[n]) < (beta * beta * dist2);

        if (fully_inside && far_enough)
        {
            scalar_t s, dphi_dt;
            eval_window<scalar_t>(dist2, inv_delta, s, dphi_dt);

            scalar_t an[3] = {
                cell_an[n][0],
                cell_an[n][1],
                cell_an[n][2]
            };

            add_grad_contrib<scalar_t>(
                dvec,
                an,
                scalar_t(1),
                dist2,
                s,
                dphi_dt,
                inv_delta,
                field_acc,
                grad_acc
            );

            continue;
        }

        if (children[n][0] < 0)
        {
            const int64_t start = leaf_offs[n];
            const int64_t end = leaf_offs[n + 1];

            for (int64_t k = start; k < end; ++k)
            {
                const int64_t pid = leaf_idx[k];
                if (pid > curT) continue;

                sub3<scalar_t>(&points[pid][0], q, dvec);

                scalar_t d2 = dot3<scalar_t>(dvec, dvec) + scalar_t(1e-8);

                scalar_t s, dphi_dt;
                eval_window<scalar_t>(d2, inv_delta, s, dphi_dt);

                scalar_t nrm[3] = {
                    normals[pid][0],
                    normals[pid][1],
                    normals[pid][2]
                };

                add_grad_contrib<scalar_t>(
                    dvec,
                    nrm,
                    areas[pid],
                    d2,
                    s,
                    dphi_dt,
                    inv_delta,
                    field_acc,
                    grad_acc
                );
            }
        }
        else
        {
            #pragma unroll
            for (int oc = 0; oc < 8; ++oc)
            {
                int64_t cid = children[n][oc];
                if (cid >= 0 && sp < WN_STACK_SIZE)
                    stack[sp++] = static_cast<int>(cid);
            }
        }
    }

    const scalar_t inv4pi = static_cast<scalar_t>(0.07957747154594767);

    out_field[b][m] = field_acc * inv4pi;

    // paper omits the 1/(4pi) factor for the gradient
    out_grad[b][m][0] = grad_acc[0] * inv4pi;
    out_grad[b][m][1] = grad_acc[1] * inv4pi;
    out_grad[b][m][2] = grad_acc[2] * inv4pi;
}

std::vector<torch::Tensor> fast_wn_clouds_cu(

    const torch::Tensor& queries,
    const torch::Tensor& curr_ts,

    const torch::Tensor& points,
    const torch::Tensor& normals,
    const torch::Tensor& areas,

    const torch::Tensor& leaf_offsets,
    const torch::Tensor& leaf_indices,

    const torch::Tensor& children,
    const torch::Tensor& cell_centres,
    const torch::Tensor& cell_radii,
    const torch::Tensor& cell_normals,
    const torch::Tensor& cell_time_iv,
    float beta,
    const float inv_delta
){
    TORCH_CHECK(queries.is_cuda(), "queries must reside on CUDA device");

    const int B = queries.size(0);
    const int M = queries.size(1);
    const int64_t total_queries = B * M;

    auto out = torch::empty({B, M}, queries.options());
    auto grads = torch::empty({B, M, 3}, queries.options());

    const dim3 threads(256);
    const dim3 blocks((total_queries + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(queries.scalar_type(), "fast_wn_kernel", ([&]
    {
        fast_wn_kernel<scalar_t><<<blocks, threads>>>(

            queries.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            curr_ts.packed_accessor32<int64_t,1,torch::RestrictPtrTraits>(),

            children.packed_accessor32<int64_t,2,torch::RestrictPtrTraits>(),
            cell_centres.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            cell_radii.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            cell_normals.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            cell_time_iv.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),

            leaf_offsets.packed_accessor32<int64_t,1,torch::RestrictPtrTraits>(),
            leaf_indices.packed_accessor32<int64_t,1,torch::RestrictPtrTraits>(),

            points.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            normals.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            areas.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),

            static_cast<scalar_t>(beta),
            static_cast<scalar_t>(inv_delta),
            out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            grads.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>()
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "launch failed: " << cudaGetErrorString(err) << '\n';

    // cudaDeviceSynchronize();

    return {out, grads};
}
