# CUDA submodules

These extensions provide octree winding-number queries, iso-contour extraction
and connection, and point-cloud distance checks for this project.

| Directory | Python import | Purpose |
| --- | --- | --- |
| [`octree_fwn`](octree_fwn) | `octree_clouds_wn` | Time-varying winding-number fields and gradients, filtered by point index at each timestep. |
| [`octree_nt_fwn`](octree_nt_fwn) | `octree_nt_clouds_wn` | Spatial winding-number fields and gradients without timestep filtering. |
| [`contour_cuda`](contour_cuda) | `qupkg.extract_isocontours` | Iso-contour extraction on triangle meshes and intersections of scalar fields. |
| [`contour_connect`](contour_connect) | `contour_connect` | CUDA neighbor search/scoring and native C++ graph tracing to connect projected iso-points into ordered contours. |
| [`simple_dists`](simple_dists) | `simple_dists` | Point-to-point distance threshold checks used to select off-surface UDF training samples. |

## Build and install

From the project root, install all five modules:

```bash
python -m pip install --no-build-isolation \
    ./submodules/octree_fwn \
    ./submodules/octree_nt_fwn \
    ./submodules/contour_cuda \
    ./submodules/contour_connect \
    ./submodules/simple_dists
```

`contour_connect` is used by `toolpath_utils.reconstruct_iso_contours`;
it complements rather than replaces mesh-based extraction in `contour_cuda`.
Rebuild it after changes to its CUDA or C++ sources.
