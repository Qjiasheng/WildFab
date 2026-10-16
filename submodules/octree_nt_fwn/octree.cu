#include <torch/extension.h>
#include "includes/utils.h"          // Int8 = std::array<int,8>
                                     // IntVec= std::vector<int>
#include <vector>
#include <stack>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include <stdexcept>
// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static constexpr double EPS = 1e-8;   // numerical safety

static inline int octant_of(const torch::Tensor& p,
                            const torch::Tensor& c)
{
    const auto ge = (p >= c).to(torch::kInt32);
    return (ge[0].item<int>()      ) |
           (ge[1].item<int>() << 1 ) |
           (ge[2].item<int>() << 2 );
}

static inline torch::Tensor child_centre(const torch::Tensor& parent,
                                         float h, int oc)
{
    const float sx = (oc & 1) ?  1.f : -1.f;
    const float sy = (oc & 2) ?  1.f : -1.f;
    const float sz = (oc & 4) ?  1.f : -1.f;
    return parent + torch::tensor({sx, sy, sz}) * (h * .5f);
}

// ---------------------------------------------------------------------------
// node structure used only while building
// ---------------------------------------------------------------------------
struct Node {
    Int8          child;
    IntVec        points;
    torch::Tensor centre;
    float         width;
};

// ---------------------------------------------------------------------------
// build_octree
// ---------------------------------------------------------------------------
std::tuple<std::vector<IntVec>, torch::Tensor>
build_octree_cu(const torch::Tensor& points_in,
                const int&  max_points_per_leaf,
                const int&  max_depth)
{
    TORCH_CHECK(points_in.dim() == 2 && points_in.size(1) == 3,
                "points must be [N,3]");

    // ---- work on a CPU copy ---------------------------------------------
    torch::Tensor points = points_in.to(torch::kCPU);

    // ---- root AABB -------------------------------------------------------
    auto lo = std::get<0>(points.min(0));
    auto hi = std::get<0>(points.max(0));
    auto c0 = (lo + hi) * 0.5f;
    float w0 = (hi - lo).max().item<float>();

    // ---- node array ------------------------------------------------------
    std::vector<Node> nodes;  nodes.reserve(1024);

    Node root;
    root.child.fill(-1);
    root.points.resize(points.size(0));
    std::iota(root.points.begin(), root.points.end(), 0);
    root.centre = c0;
    root.width  = w0;
    nodes.emplace_back(std::move(root));

    // ---- DFS subdivision -------------------------------------------------
    struct Job { int id, depth; };
    std::stack<Job> st;  st.push({0,0});

    while (!st.empty())
    {
        Job job = st.top(); st.pop();
        int parent = job.id, depth = job.depth;

        if (nodes[parent].points.size() <= (size_t)max_points_per_leaf ||
            depth >= max_depth)
            continue;

        float h = nodes[parent].width * 0.5f;
        int first_child = (int)nodes.size();

        // create children
        for (int oc=0; oc<8; ++oc) {
            Node child;
            child.child.fill(-1);
            child.centre = child_centre(nodes[parent].centre, h, oc);
            child.width  = h;
            nodes.emplace_back(std::move(child));
            nodes[parent].child[oc] = first_child + oc;
        }

        // distribute points
        for (int pid : nodes[parent].points) {
            int oc = octant_of(points[pid], nodes[parent].centre);
            nodes[nodes[parent].child[oc]].points.push_back(pid);
        }
        
        // compute min/max for children and push to stack
        for (int oc = 0; oc < 8; ++oc)
        {
            int cid = nodes[parent].child[oc]; 
            st.push({cid, depth + 1});
        }
        nodes[parent].points.clear();
    }

    // ---- convert to public tensors --------------------------------------
    int64_t M = (int64_t)nodes.size();

    // 1. point indices (vector<vector<int>>)
    std::vector<IntVec> point_indices(M);
    for (int64_t i=0;i<M;++i) point_indices[i] = std::move(nodes[i].points);

    // int32 since number of nodes can exceed int16 range sometimes
    torch::Tensor children =
        torch::full({M, 8}, -1,
                    torch::TensorOptions().dtype(torch::kInt32)
                                           .device(torch::kCPU));

    for (int64_t i=0;i<M;++i)
        for (int oc=0; oc<8; ++oc)
            children.index_put_({i,oc}, nodes[i].child[oc]);

    return {std::move(point_indices), children}; 
}





/* ---------- helper ---------------------------------------------------- */
inline bool is_leaf(int64_t nid,
                    const torch::Tensor& children) noexcept
{
    return children[nid][0].item<int64_t>() == -1;
}

/* ---------- PASS #1 : centres & (unnormalised) normal ----------------- */
struct Agg
{
    double        a_sum{0.0};     // ∑ area
    torch::Tensor p_sum;          // ∑ area ⋅ position   (shape [3])
    torch::Tensor n_sum;          // ∑ area ⋅ normal     (shape [3])

    Agg(const torch::TensorOptions& opt)
        : p_sum(torch::zeros({3}, opt)),
          n_sum(torch::zeros({3}, opt)) {}
};

static Agg aggregate_rec(int64_t nid,
                         const torch::Tensor&  points,
                         const torch::Tensor&  normals,
                         const torch::Tensor&  areas,
                         const std::vector<IntVec>& plist,
                         const torch::Tensor&  children,
                         torch::Tensor& centres,
                         torch::Tensor& nhat)
{
    const auto opts = points.options();
    Agg S(opts);

    if (is_leaf(nid, children))
    {
        /* ------- gather this leaf’s points ---------------------------- */
        const IntVec& ids = plist[nid];
        const int64_t K   = ids.size();
        if (K == 0) {                             // completely empty leaf
            centres[nid].zero_();
            nhat   [nid].zero_();
            return S;
        }

        torch::Tensor idx =
            torch::from_blob(const_cast<int*>(ids.data()), {K},
                             torch::TensorOptions().dtype(torch::kInt32))
            .to(torch::kInt64)
            .clone();             // own a copy → safe after we return

        torch::Tensor a = areas  .index_select(0, idx).unsqueeze(1); // [K,1]
        torch::Tensor p = points .index_select(0, idx);             // [K,3]
        torch::Tensor n = normals.index_select(0, idx);             // [K,3]

        S.a_sum = a.sum().item<double>();
        S.p_sum = (a * p).sum(0);
        S.n_sum = (a * n).sum(0);
    }
    else            /* internal node → recurse through its 8 children */
    {
        for (int oc = 0; oc < 8; ++oc)
        {
            int64_t cid = children[nid][oc].item<int64_t>();
            Agg C = aggregate_rec(cid, points, normals, areas,
                                   plist, children, centres, nhat);
            S.a_sum += C.a_sum;
            S.p_sum += C.p_sum;
            S.n_sum += C.n_sum;
        }
    }

    /* ---------- write results for this node --------------------------- */
    const double denom = S.a_sum + EPS;
    centres.index_put_({nid}, S.p_sum / denom);
    nhat   .index_put_({nid}, S.n_sum);          // keep un-normalised
    return S;
}

/* ---------- PASS #2 : exact radius ----------------------------------- */
static double max_dist_rec(int64_t nid,
                           const torch::Tensor&  points,
                           const std::vector<IntVec>& plist,
                           const torch::Tensor&  children,
                           const torch::Tensor&  centres,
                           const torch::Tensor&  c_here)
{
    double r = 0.0;

    if (is_leaf(nid, children))
    {
        const IntVec& ids = plist[nid];
        if (ids.empty()) return 0.0;

        torch::Tensor idx =
            torch::from_blob(const_cast<int*>(ids.data()), {(int64_t)ids.size()},
                             torch::TensorOptions().dtype(torch::kInt32))
            .to(torch::kInt64).clone();

        torch::Tensor p = points.index_select(0, idx);
        torch::Tensor d = (p - c_here).pow(2).sum(1).sqrt();
        r = d.max().item<double>();
    }
    else
    {
        for (int oc = 0; oc < 8; ++oc)
        {
            int64_t cid = children[nid][oc].item<int64_t>();
            r = std::max(r,
                         max_dist_rec(cid, points, plist,
                                      children, centres, c_here));
        }
    }
    return r;
}

/* ---------- PUBLIC INITIALISER --------------------------------------- */
std::vector<torch::Tensor> initialize_octree_cu(
        const torch::Tensor&  points_in,
        const torch::Tensor&  normals_in,
        const torch::Tensor&  areas_in,
        const std::vector<IntVec>& plist,
        const torch::Tensor&  children_in)
{
    TORCH_CHECK(points_in.sizes() == normals_in.sizes(),
                "normals must have same shape as points");
    TORCH_CHECK(areas_in.size(0) == points_in.size(0),
                "areas length mismatch");

    /* work on CPU copies for simplicity -------------------------------- */
    const auto dtype = points_in.dtype();
    const auto dev   = points_in.device();

    torch::Tensor points   = points_in .cpu();
    torch::Tensor normals  = normals_in.cpu();
    torch::Tensor areas    = areas_in  .cpu();
    torch::Tensor children = children_in.to(torch::kInt64).cpu();

    const int64_t M = children.size(0);
    torch::Tensor centres = torch::zeros({M,3}, dtype);
    torch::Tensor nhat    = torch::zeros({M,3}, dtype);
    torch::Tensor radii   = torch::zeros({M  }, dtype);

    /* ---------- pass #1 ------------------------------------------------ */
    aggregate_rec(0, points, normals, areas,
                  plist, children, centres, nhat);

    /* ---------- pass #2 ------------------------------------------------ */
    for (int64_t nid = 0; nid < M; ++nid)
    {
        double r = max_dist_rec(nid, points, plist,
                                children, centres, centres[nid]);
        radii[nid] = r;
    }

    /* move back to original device ------------------------------------ */
    if (!dev.is_cpu())
    {
        centres = centres.to(dev);
        nhat    = nhat   .to(dev);
        radii   = radii  .to(dev);
    }
    return {centres, nhat, radii};
}