#define MAX_NUM_CU 4
// using MyUint1 = ac_int<1, false>;
#include "unrolled_loop.hpp"
int numRows,numCols,numNonz;

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


#define DEBUG(x) std::cout <<" : "<< x << std::endl;
//Structure to hold a node information
struct ComputeUnit
{
  
  MyUint1 *usm_mask;


};
//Structure to hold a node information
struct HostGraphData
{
  
  // std::vector<unsigned int> h_graph_nodes_edges;
  std::vector<MyUint1> h_graph_mask;
  


};

typedef std::array<HostGraphData, MAX_NUM_CU> GraphData;


void HostGraphDataGenerate(int indexPE,int start_vertex,GraphData &fpga_cu_data,std::vector<unsigned int>&source_meta,std::vector<unsigned int>&source_indptr,std::vector<unsigned int>&source_inds,std::vector<unsigned int>& old_buffer_size_meta,
	std::vector<unsigned int>& old_buffer_size_indptr,
	std::vector<unsigned int>& old_buffer_size_inds) 
{

  numRows  = source_meta[0 + old_buffer_size_meta[indexPE]];  // this it the value we want! (rows)
	numNonz  = source_meta[2 + old_buffer_size_meta[indexPE]];  // nonZ count -> total edges
  // Sanity Check if we loaded the graph properly
  assert(numRows <= numCols);

  std::cout << std::setw(6) << std::left << "# Graph Information" << "\n Vertices (nodes) = " << numRows << " \n Edges = "<< numNonz << "\n";
	
  
  
  // allocate host memory


  fpga_cu_data[indexPE].h_graph_mask.resize(numCols);

 // initialise all the values to 0
  std::fill(fpga_cu_data[indexPE].h_graph_mask.begin(), fpga_cu_data[indexPE].h_graph_mask.end(), 0);  







    

   fpga_cu_data[indexPE].h_graph_mask[start_vertex]=1; 

}


// initialize device arr with val, if needed set arr[pos] = pos_val
template <typename T>
void initialize(queue &Q,T val, T *arr,int gws,int pos = -1, T pos_val = -1)

{


    Q.parallel_for(gws, [=](id<1> i) [[intel::kernel_args_restrict]] {
                                  
                                      arr[i] = val;
    
                                      if (i == pos)
                                      {
                                          arr[pos] = pos_val;
                                      }
                                   }).wait(); 

}

// Convention 
// CapitalCamelCase - pointer

class Matrix{
public:
  unsigned int nodeCount; // number of nodes(vertices)
  unsigned int edgeCount; // number of edges(connections)
  unsigned int source; // the begining node
  std::vector<unsigned int> Offset; // indptr from old convention this stores #totalEdgesforX = rowPointer[x+1] - rowPointer[x]
  std::vector<unsigned int> Position; // inds from old convention this stores the data we want colIndices[rowPointer[x]] till colIndices[] 
  std::vector<char> VisitMask; // old h_updating_mask for usm_updating_mask (newly generated visited)
  std::vector<char> Visit; // old h_visited (old visited data)
  std::vector<int>  Distance; // h_dist for distances from source
  std::vector<unsigned int> Frontier; // pipe data
  void Populate(unsigned int s,unsigned int v, unsigned int e,
                std::vector<unsigned int>& source_indptr,std::vector<unsigned int>& source_inds,std::vector<int>& h_dist,std::vector<char> &h_updating_graph_mask,std::vector<char> &h_graph_visited,std::vector<unsigned int> &h_graph_pipe){
    source = s;
    nodeCount = v;
    edgeCount = e;
    Offset = source_indptr;
    Position = source_inds;
    VisitMask = h_updating_graph_mask;
    Visit = h_graph_visited;
    Distance = h_dist;
    Frontier = h_graph_pipe;


  }
};
