#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>

struct Edge { bool one_way; float score; int64_t a, b; };
struct LevelGraph { std::vector<std::vector<int64_t>> chains; std::vector<uint8_t> closed; int64_t isolated = 0; };

// Preserve the Python greedy policy, but run independent levels on native CPU workers.
std::vector<torch::Tensor> trace_graph_batch(torch::Tensor points, torch::Tensor tangents, torch::Tensor ids, torch::Tensor scores, torch::Tensor offsets, int64_t min_points) {
    for (const auto& t : {points, tangents, ids, scores, offsets}) TORCH_CHECK(t.device().is_cpu() && t.is_contiguous(), "graph inputs must be contiguous CPU tensors");
    TORCH_CHECK(points.scalar_type()==torch::kFloat32 && tangents.scalar_type()==torch::kFloat32 && scores.scalar_type()==torch::kFloat32 && ids.scalar_type()==torch::kInt64 && offsets.scalar_type()==torch::kInt64, "invalid graph input dtype");
    const auto N = points.size(0);
    TORCH_CHECK(points.dim()==2 && points.size(1)==3 && tangents.sizes()==points.sizes(), "geometry must be Nx3");
    TORCH_CHECK(ids.dim()==3 && ids.size(0)==N && ids.size(1)==2 && scores.sizes()==ids.sizes(), "candidates must be Nx2xK");
    TORCH_CHECK(offsets.dim()==1 && offsets.numel()>=1 && min_points>=2, "invalid offsets/min_points");
    const auto* off=offsets.data_ptr<int64_t>();
    const int64_t L=offsets.numel()-1, K=ids.size(2)*2;
    TORCH_CHECK(off[0]==0 && off[L]==N, "offsets must cover all points");
    for(int64_t l=0;l<L;++l) TORCH_CHECK(off[l]<=off[l+1], "offsets must be monotone");
    const auto* p=points.data_ptr<float>(); const auto* t=tangents.data_ptr<float>();
    const auto* cand=ids.data_ptr<int64_t>(); const auto* cost=scores.data_ptr<float>();
    std::vector<LevelGraph> output(L);
    at::parallel_for(0,L,1,[&](int64_t begin,int64_t end){
        for(int64_t l=begin;l<end;++l){
            const int64_t base=off[l], n=off[l+1]-base;
            std::unordered_map<int64_t,float> proposals; proposals.reserve(n*K);
            for(int64_t i=0;i<n;++i) for(int64_t k=0;k<K;++k){
                int64_t j=cand[(base+i)*K+k]; float s=cost[(base+i)*K+k];
                if(j<0 || !std::isfinite(s)) continue;
                TORCH_CHECK(j>=base && j<base+n, "candidate crossed iso boundary");
                j-=base; int64_t key=i*n+j;
                auto result=proposals.emplace(key,s);
                if(!result.second) result.first->second=std::min(result.first->second,s);
            }
            std::vector<Edge> edges; edges.reserve(proposals.size());
            for(auto entry:proposals){
                const int64_t i=entry.first/n,j=entry.first%n;
                auto reverse=proposals.find(j*n+i);
                if(i>j && reverse!=proposals.end()) continue;
                edges.push_back({reverse==proposals.end(), reverse==proposals.end()?entry.second:std::max(entry.second,reverse->second), std::min(i,j),std::max(i,j)});
            }
            std::sort(edges.begin(),edges.end(),[](const Edge& a,const Edge& b){
                if(a.one_way!=b.one_way)return a.one_way<b.one_way;
                if(a.score!=b.score)return a.score<b.score;
                if(a.a!=b.a)return a.a<b.a;
                return a.b<b.b;
            });
            std::vector<int64_t> slots(2*n,-1),parent(n),size(n,1);
            std::iota(parent.begin(),parent.end(),0);
            auto root=[&](int64_t i){while(parent[i]!=i){parent[i]=parent[parent[i]];i=parent[i];}return i;};
            for(auto edge:edges){
                int64_t i=edge.a,j=edge.b;float di=0,dj=0;
                for(int a=0;a<3;++a){float d=p[3*(base+j)+a]-p[3*(base+i)+a];di+=d*t[3*(base+i)+a];dj-=d*t[3*(base+j)+a];}
                int si=di>=0,sj=dj>=0;
                if(slots[2*i+si]>=0 || slots[2*j+sj]>=0)continue;
                auto ri=root(i),rj=root(j);
                if(ri==rj && size[ri]<std::max(int64_t(3),min_points))continue;
                slots[2*i+si]=j;slots[2*j+sj]=i;
                if(ri!=rj){parent[rj]=ri;size[ri]+=size[rj];}
            }
            std::vector<uint8_t> visited(n,0);
            for(int64_t i=0;i<n;++i)if(slots[2*i]<0 && slots[2*i+1]<0)++output[l].isolated;
            for(int degree=1;degree<=2;++degree)for(int64_t start=0;start<n;++start){
                if(visited[start] || (int(slots[2*start]>=0)+int(slots[2*start+1]>=0))!=degree)continue;
                int64_t current=start,previous=-1; bool closed=false; std::vector<int64_t> chain;
                while(!visited[current]){
                    visited[current]=1;chain.push_back(base+current);int64_t next=-1;
                    for(int a=0;a<2;++a)if(slots[2*current+a]>=0 && slots[2*current+a]!=previous){next=slots[2*current+a];break;}
                    if(next<0)break;
                    if(next==start){closed=true;break;}
                    previous=current;current=next;
                }
                if(int64_t(chain.size())>=min_points){if(closed)chain.push_back(base+start);output[l].chains.push_back(std::move(chain));output[l].closed.push_back(closed);}
            }
        }
    });
    int64_t chain_count=0,point_count=0;
    for(auto& level:output)for(auto& chain:level.chains){++chain_count;point_count+=chain.size();}
    auto opts=torch::TensorOptions().dtype(torch::kInt64);
    auto indices=torch::empty({point_count},opts), contour_offsets=torch::empty({chain_count+1},opts), iso_offsets=torch::empty({L+1},opts), closed=torch::empty({chain_count},opts.dtype(torch::kBool)), isolated=torch::empty({L},opts);
    int64_t ci=0,pi=0;contour_offsets.data_ptr<int64_t>()[0]=0;iso_offsets.data_ptr<int64_t>()[0]=0;
    for(int64_t l=0;l<L;++l){
        for(size_t j=0;j<output[l].chains.size();++j){auto& chain=output[l].chains[j];std::copy(chain.begin(),chain.end(),indices.data_ptr<int64_t>()+pi);pi+=chain.size();closed.data_ptr<bool>()[ci]=output[l].closed[j];contour_offsets.data_ptr<int64_t>()[++ci]=pi;}
        iso_offsets.data_ptr<int64_t>()[l+1]=ci;isolated.data_ptr<int64_t>()[l]=output[l].isolated;
    }
    return {indices,contour_offsets,iso_offsets,closed,isolated};
}
