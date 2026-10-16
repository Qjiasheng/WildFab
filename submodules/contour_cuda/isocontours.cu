#include <torch/extension.h>
#include "includes/utils.h"
#include <iostream>
#include <cstdio>
#include <cmath>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>


// CUDA kernel for extracting iso-contour segments
// marching triangles at each iso-value yields point pairs
// shared edges have repeated interpolated points, then merge 
template <typename scalar_t>
__global__ void extract_contours_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> vertices,
    const torch::PackedTensorAccessor<int32_t, 2, torch::RestrictPtrTraits> faces,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits> fields,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits> iso_values,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> contour_points,
    torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits> pairpoint_nums,
    torch::PackedTensorAccessor<int64_t, 3, torch::RestrictPtrTraits> endpoint_keys,
    const int num_faces,
    const int num_iso_values)
{
    // parallel along triangle faces
    int face_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_id >= num_faces) return;

    // Load vertex indices for this face
    int v0_idx = faces[face_id][0];
    int v1_idx = faces[face_id][1];
    int v2_idx = faces[face_id][2];

    // Load field values at each vertex for the current face
    scalar_t f0 = fields[v0_idx];
    scalar_t f1 = fields[v1_idx];
    scalar_t f2 = fields[v2_idx];

    // Iterate over iso-values to find contour points
    for (int iso_id = 0; iso_id < num_iso_values; ++iso_id) {
        scalar_t iso_value = iso_values[iso_id];

        // Half-open classification: equality always belongs to the upper side.
        // This selects exactly two edges, including at a vertex-valued level.
        const int ids[3] = {v0_idx, v1_idx, v2_idx};
        const scalar_t values[3] = {f0, f1, f2};
        const bool below[3] = {f0 < iso_value, f1 < iso_value, f2 < iso_value};
        if (below[0] == below[1] && below[1] == below[2]) continue;

        scalar_t points[6];
        int64_t keys[2];
        int point_count = 0;
        for (int edge = 0; edge < 3; ++edge) {
            const int next = (edge + 1) % 3;
            if (below[edge] == below[next]) continue;
            // Canonical edge orientation gives identical interpolation on both faces.
            const bool ordered = ids[edge] < ids[next];
            const int a = ordered ? ids[edge] : ids[next];
            const int b = ordered ? ids[next] : ids[edge];
            const scalar_t fa = ordered ? values[edge] : values[next];
            const scalar_t fb = ordered ? values[next] : values[edge];
            const scalar_t t = (iso_value - fa) / (fb - fa);
            for (int k = 0; k < 3; ++k) {
                points[3 * point_count + k] =
                    vertices[a][k] + t * (vertices[b][k] - vertices[a][k]);
            }
            // Keep edge identity even when t is 0 or 1: zero-length segments
            // preserve connectivity around vertices without geometric ambiguity.
            keys[point_count++] = int64_t(a) * vertices.size(0) + b;
        }
        const int idx = atomicAdd(&pairpoint_nums[iso_id], 1);
        for (int k = 0; k < 6; ++k) contour_points[iso_id][idx][k] = points[k];
        endpoint_keys[iso_id][idx][0] = keys[0];
        endpoint_keys[iso_id][idx][1] = keys[1];
    }
}

// CUDA kernel for merging point pairs, multiple subcontours in 2nd dim, but insert -1e9 as separator
// Subcontours are ordered; closed contours repeat their starting point. 
// subcontours are separated by -1e9, NOTE even single contour has a separator at the end.
// [subcontour1, -1e9, 00000...], [subcontour1, -1e9, subcontour2, -1e9, 00000...]
template <typename scalar_t>
__global__ void merge_segments_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> contour_points,
    const torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits> pairpoint_nums,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> contours_merged,
    torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits> contour_nums,

    const torch::PackedTensorAccessor<int64_t, 3, torch::RestrictPtrTraits> endpoint_keys,
    uint8_t* used_global, // use global memory
    const int num_faces,
    const int num_iso_values)
{
    int iso_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (iso_id >= num_iso_values) return;

    int pair_num = pairpoint_nums[iso_id];
    if (pair_num == 0) return; // no pairs, no need to merge

    // contour_points [num_isovalues, num_faces, 6] stores pairs of intersection points across all triangles 
    // pairpoint_nums [num_isovalues] stores how many pairs are extracted at each iso-value

    // use global memory to store used flags, get current thread used.
    uint8_t* used = used_global + int64_t(iso_id) * num_faces;
    
    // merge pairs
    int current_contour_index = 0;
    int contours_found = 0;

 

    for (int face_id = 0; face_id < pair_num; ++face_id) {
        if (used[face_id]) continue;

        // start a new contour and set a separator at (x, y, z) = (-1e9, -1e9, -1e9)
        // if more than one subcontours wrt iso-value
        if (contours_found > 0) {
            for (int i = 0; i < 3; ++i) { 
                contours_merged[iso_id][current_contour_index][i] = -1e9; // Separator marker
            }
            current_contour_index++;
        }

        const int contour_start = current_contour_index;
        int64_t first_key = endpoint_keys[iso_id][face_id][0];
        int64_t last_key = endpoint_keys[iso_id][face_id][1];

        for (int i = 0; i < 3; ++i)
            contours_merged[iso_id][current_contour_index][i] = contour_points[iso_id][face_id][i];          // P0
        ++current_contour_index;

        for (int i = 0; i < 3; ++i) {
            scalar_t p = contour_points[iso_id][face_id][3 + i]; // P1
            contours_merged[iso_id][current_contour_index][i] = p;

        }
        ++current_contour_index;

        used[face_id] = true;
        ++contours_found;

        // Extend one end, then reverse once to grow the other end of an open
        // chain. Stop at closure so a branch cannot consume another loop.
        bool reversed = false;
        while (last_key != first_key) {
            int candidate = -1;
            int other_end = 0;
            for (int next_face_id = 0; next_face_id < pair_num; ++next_face_id) {
                if (used[next_face_id]) continue;
                if (endpoint_keys[iso_id][next_face_id][0] == last_key) {
                    candidate = next_face_id;
                    other_end = 1;
                    break;
                }
                if (endpoint_keys[iso_id][next_face_id][1] == last_key) {
                    candidate = next_face_id;
                    other_end = 0;
                    break;
                }
            }
            if (candidate < 0) {
                if (reversed) break;
                for (int left = contour_start, right = current_contour_index - 1;
                     left < right; ++left, --right) {
                    for (int k = 0; k < 3; ++k) {
                        scalar_t tmp = contours_merged[iso_id][left][k];
                        contours_merged[iso_id][left][k] = contours_merged[iso_id][right][k];
                        contours_merged[iso_id][right][k] = tmp;
                    }
                }
                const int64_t tmp = first_key;
                first_key = last_key;
                last_key = tmp;
                reversed = true;
                continue;
            }
            for (int k = 0; k < 3; ++k) {
                contours_merged[iso_id][current_contour_index][k] =
                    contour_points[iso_id][candidate][3 * other_end + k];
            }
            ++current_contour_index;
            last_key = endpoint_keys[iso_id][candidate][other_end];
            used[candidate] = true;
        }

    }
    // Add a final separator after the last contour, even a single contour
    if (contours_found > 0) {
        for (int i = 0; i < 3; ++i) {
            contours_merged[iso_id][current_contour_index][i] = -1e9;
        }
        current_contour_index++;
    }
    // how many contours for one iso-value
    contour_nums[iso_id] = contours_found;
}


std::vector<torch::Tensor> extract_isocontours_cu(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& fields,
    const torch::Tensor& iso_values)
{
    // iso_contours[iso_value] = [contour1, contour2[subcontour1, subcontour2], ...]

    const c10::cuda::CUDAGuard device_guard(vertices.device());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    int num_faces = faces.size(0);
    int num_iso_values = iso_values.size(0);

    // std::cout << "num faces " << num_faces << "\n" << "num isovalues "<< num_iso_values << std::endl;

    // every iso-value has num_faces space to store point pairs, even no intersections at an iso-value
    torch::Tensor contour_points = torch::zeros({num_iso_values, num_faces, 6}, vertices.options());
    // for each iso-value, how many contours are extracted
    torch::Tensor pairpoint_nums = torch::zeros({num_iso_values}, torch::TensorOptions().dtype(torch::kInt32).device(vertices.device()));
    torch::Tensor contours_merged = torch::zeros({num_iso_values, 0, 3}, vertices.options());
    // how many contours are extracted at each iso-value
    torch::Tensor contour_nums = torch::zeros({num_iso_values}, torch::TensorOptions().dtype(torch::kInt32).device(vertices.device()));

    if (num_faces == 0 || num_iso_values == 0) return {contours_merged, contour_nums};
    auto endpoint_keys = torch::empty({num_iso_values, num_faces, 2},
                                      vertices.options().dtype(torch::kInt64));
    auto used = torch::zeros({num_iso_values, num_faces},
                             vertices.options().dtype(torch::kUInt8));
    int threads = 256;
    int blocks = (num_faces + threads - 1) / threads;
    int blocks_merge = (num_iso_values + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(vertices.scalar_type(), "extract_isocontours_parallel", 
    ([&] {
        extract_contours_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            vertices.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            faces.packed_accessor<int32_t, 2, torch::RestrictPtrTraits>(),
            fields.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits>(),
            iso_values.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits>(),
            contour_points.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            pairpoint_nums.packed_accessor<int32_t, 1, torch::RestrictPtrTraits>(),
            endpoint_keys.packed_accessor<int64_t, 3, torch::RestrictPtrTraits>(),

            num_faces, num_iso_values
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // S segments in C chains need S + 2*C rows (start + separator).
        // C <= S, so 3*max(S) is safe without allocating 3*num_faces rows.
        // Reading the count synchronizes extraction before allocating its output.
        const int64_t max_segments = pairpoint_nums.max().item<int>();
        contours_merged = torch::zeros({num_iso_values, 3 * max_segments, 3}, vertices.options());

        merge_segments_kernel<scalar_t><<<blocks_merge, threads, 0, stream>>>(
            contour_points.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            pairpoint_nums.packed_accessor<int32_t, 1, torch::RestrictPtrTraits>(),
            contours_merged.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            contour_nums.packed_accessor<int32_t, 1, torch::RestrictPtrTraits>(),
            endpoint_keys.packed_accessor<int64_t, 3, torch::RestrictPtrTraits>(),
            used.data_ptr<uint8_t>(),
            num_faces, num_iso_values
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

    }));

    return {contours_merged, contour_nums};
}


