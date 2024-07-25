#define MAX_NUM_CU 4
// using MyUint1 = ac_int<1, false>;
#include "unrolled_loop.hpp"
int numRows = 0,numCols =0,numNonz =0;

constexpr int log2(int num) {
    int result = 0;
    int running = num;

    while (running > 1) {
        result++;
        running /= 2;
    }

    int comp = 1;

    for (int i = 0; i < result; i++) {
        comp *= 2;
    }

    if (num != comp) {
        result++;
    }
    
    return result;
}

constexpr int BUFFER_SIZE = 16;
using MyUint1 = char; 
using d_type3 = char;
using Uint32 = unsigned int;


class Timer {
public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

// Convention 
// CapitalCamelCase - pointer

// class Matrix{
// public:
//   unsigned int nodeCount; // number of nodes(vertices)
//   unsigned int edgeCount; // number of edges(connections)
//   unsigned int source; // the begining node
//   std::vector<unsigned int> Offset; // indptr from old convention this stores #totalEdgesforX = rowPointer[x+1] - rowPointer[x]
//   std::vector<unsigned int> Position; // inds from old convention this stores the data we want colIndices[rowPointer[x]] till colIndices[] 
//   std::vector<char> VisitMask; // old h_updating_mask for usm_updating_mask (newly generated visited)
//   std::vector<char> Visit; // old h_visited (old visited data)
//   std::vector<int>  Distance; // h_dist for distances from source
//   std::vecotr<unsigned int> Frontier; // 
//   void Populate(unsigned int s,unsigned int v, unsigned int e,
//                 std::vector<unsigned int>& source_indptr,std::vector<unsigned int>& source_inds,std::vector<int>& h_dist,std::vector<char> &h_updating_graph_mask,std::vector<char> &h_graph_visited){
//     source = s;
//     nodeCount = v;
//     edgeCount = e;
//     Offset = source_indptr;
//     Position = source_inds;
//     VisitMask = h_updating_graph_mask;
//     Visit = h_graph_visited;
//     Distance = h_dist;


//   }
// };

