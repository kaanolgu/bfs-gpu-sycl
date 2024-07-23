#include <cstdlib>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <random>
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"
#include "run_bfs_gpu.hpp"
#include "run_bfs_cpu.hpp"
#include "graph_load.hpp"

using namespace std;
namespace ext_oneapi = sycl::ext::oneapi;
Uint32 matrixID;

// [ext_oneapi_cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.2]

    const char separator    = ' ';
    const int nameWidth     = 24;
    const int numWidth      = 24;

//-------------------------------------------------------------------
//--initialize array with maximum limit
//-------------------------------------------------------------------
template<typename datatypeA,typename datatypeB>
void print_levels(std::vector<datatypeA> &A,std::string nameA,std::vector<datatypeB> &B ,std::string nameB,int size){
    printf("|---------------------------------------------------|\n");
    printf("|                    VERIFY RESULTS                 |\n");
    printf("|-------------------------+-------------------------|\n");
    std::cout <<"|                     "+ nameA +" | " + nameB+"                    |\n";
    printf("|-------------------------+-------------------------|\n");
    
  for(int i = 0; i < size; i++){
    //  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Explore_4  : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(time_kernel3) + " (s) " << "| " << std::endl;

 
     std::cout << "| " << std::left << std::setw(nameWidth/2+1) << std::setfill(separator) << "Level #" +  std::to_string(i) + " : " <<std::right << std::setw(nameWidth/2-2) << std::setfill(separator) << std::to_string(count(A.begin(), A.end(), i)) <<  " | " << std::left<< std::setw(numWidth) << std::setfill(separator)  << std::to_string(count(B.begin(), B.end(), i)) << "| " << std::endl;
  }
      printf("|-------------------------+-------------------------|\n");
  char passed = false; 
  if( equal(A.begin(), A.end(), B.begin()) ){
      passed = true; 
  }
  if (passed)
    std::cout << "| " << std::left << std::setw(nameWidth*2+2) << std::setfill(separator) << " TEST PASSED!"  <<"|"<<std::endl;
  else
    std::cout << "TEST FAILED!" << std::endl;
        printf("|---------------------------------------------------|\n");
int sum =0;
int nok_sum =0;
for(int level=0; level < size; level++){
  for (size_t i = 0; i < B.size(); ++i) {
        if (B[i] == level) {
            
            if (A[i] != level) {
              nok_sum+=1;
            //     std::cout << "ERROR at index " << i << ", A[i] = " << A[i] << ", B[i] = " << B[i] << std::endl;
            }else{
                sum +=1;
            }
        }
    }
    std::cout<<"Level "<< level <<" \nMatching -> "<< sum << std::endl;
    std::cout<<"Non Matching -> "<< nok_sum << std::endl;
}
}

int main(int argc, char * argv[])
{
  
  datasetName = argv[1];  

  int start_vertex = stoi(argv[2]);

	std::vector<Uint32> old_buffer_size_meta(1,0);
	std::vector<Uint32> old_buffer_size_indptr(1,0);
	std::vector<Uint32> old_buffer_size_inds(1,0);
	std::vector<Uint32> old_buffer_size_config(1,0);
  Uint32 offset_meta =0;
  Uint32 offset_indptr =0;
  Uint32 offset_inds =0;

  
  std::cout << "######################LOADING MATRIX#########################" << std::endl;
  loadMatrix(NUM_COMPUTE_UNITS, old_buffer_size_meta, old_buffer_size_indptr, old_buffer_size_inds,
             offset_meta, offset_indptr, offset_inds);
  std::cout << "#############################################################\n" << std::endl;
  numCols = source_meta[1];  // cols -> total number of vertices

  ////////////
  // FPGA
  ///////////

  // allocate mem for the result on host side
  std::vector<Uint32> h_dist(numCols,-2);
  h_dist[start_vertex]=0;  
  std::vector<Uint32> h_graph_nodes_start;
  //read the start_vertex node from the file
  //set the start_vertex node as 1 in the mask
  std::vector<MyUint1> h_updating_graph_mask(numCols,0);
  // make this a different datatype and cast it to the kernel
  // hpm version of the stratix 10 try 
  std::vector<MyUint1> h_graph_visited(numCols,0); 
  h_graph_visited[start_vertex]=1; 
  int indptr_end = old_buffer_size_indptr[1];
  int inds_end = old_buffer_size_inds[1];
  // initalize the memory
  int numEdges  = source_meta[2 + old_buffer_size_meta[0]];  // nonZ count -> total edges
  numRows  = source_meta[0 + old_buffer_size_meta[0]];  // this it the value we want! (rows)
  // Sanity Check if we loaded the graph properly
  assert(numRows <= numCols);
  std::cout << std::setw(6) << std::left << "# Graph Information" << "\n Vertices (nodes) = " << numRows << " \n Edges = "<< numEdges << "\n";

  FPGARun(numCols,
                  source_inds,
                  source_indptr,
                  h_updating_graph_mask,
                  h_graph_visited,
                  h_dist,
                  start_vertex,numEdges);  

  // initalize the memory again
  std::vector<Uint32> host_graph_mask(numCols,0);
  std::vector<Uint32> host_updating_graph_mask(numCols,0);
  std::vector<Uint32> host_graph_visited(numCols,0);
  // allocate mem for the result on host side
  std::vector<int> host_level(numCols,-1);
    
  //set the start_vertex node as 1 in the mask
  host_graph_mask[start_vertex]=1;
  host_graph_visited[start_vertex]=1;
  host_level[start_vertex]=0; 

  run_bfs_cpu(numCols,source_indptr,source_inds, host_graph_mask, host_updating_graph_mask, host_graph_visited, host_level);

  // Select the element with the maximum value
  auto it = std::max_element(host_level.begin(), host_level.end());
  // Check if iterator is not pointing to the end of vector
  int maxLevelCPU = (*it +2);

  print_levels(host_level,"cpu",h_dist,"fpga",maxLevelCPU); // CPU Results


  return 0;
}