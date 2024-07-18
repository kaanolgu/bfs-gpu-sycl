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
unsigned int matrixID;

// [ext_oneapi_cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.2]

#define DATA_SIZE 512
#define INDPTR_SIZE 2096648




 void setBit(unsigned int ind,std::vector<unsigned int> &m_bitmap) {
	m_bitmap[ind] = 1;
}

 bool testBit(unsigned int ind,std::vector<unsigned int> &m_bitmap) {
	return (m_bitmap[ind] == 1);
}

    const char separator    = ' ';
    const int nameWidth     = 24;
    const int numWidth      = 24;


//-------------------------------------------------------------------
//--initialize array with maximum limit
//-------------------------------------------------------------------
template<typename datatype>
void print_levels(std::vector<datatype> &A,std::string nameA,std::vector<datatype> &B ,std::string nameB,int size){
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

}




int main(int argc, char * argv[])
{
  // Measure execution times.
  // double elapsed_s = 0;
  // double elapsed_p = 0;
  
    
    
///////////////////////////////////////////////////////////////////////////
// ./build/fpga_bfs.emu rmat-19-32 #partitions

  datasetName = argv[1];  

  int start_vertex = stoi(argv[2]);

	std::vector<unsigned int> old_buffer_size_meta(1,0);
	std::vector<unsigned int> old_buffer_size_indptr(1,0);
	std::vector<unsigned int> old_buffer_size_inds(1,0);
	std::vector<unsigned int> old_buffer_size_config(1,0);
  unsigned int offset_meta =0;
  unsigned int offset_indptr =0;
  unsigned int offset_inds =0;

  std::vector<unsigned int> old_buffer_size_meta_cpu(1,0);
	std::vector<unsigned int> old_buffer_size_indptr_cpu(1,0);
	std::vector<unsigned int> old_buffer_size_inds_cpu(1,0);
	std::vector<unsigned int> old_buffer_size_config_cpu(1,0);
  unsigned int offset_meta_cpu =0;
  unsigned int offset_indptr_cpu =0;
  unsigned int offset_inds_cpu =0;


  
  std::cout << "######################LOADING MATRIX#########################" << std::endl;
  loadMatrix(NUM_COMPUTE_UNITS, old_buffer_size_meta, old_buffer_size_indptr, old_buffer_size_inds,
             offset_meta, offset_indptr, offset_inds);
  std::cout << "#############################################################\n" << std::endl;
  numCols = source_meta[1];  // cols -> total number of vertices
  loadMatrixCPU(1, old_buffer_size_meta_cpu, old_buffer_size_indptr_cpu, old_buffer_size_inds_cpu,
                offset_meta_cpu, offset_indptr_cpu, offset_inds_cpu);


////////////
// FPGA
///////////


 
    // allocate mem for the result on host side
  std::vector<int> h_dist(numCols,-1);
  h_dist[start_vertex]=0;  

std::vector<unsigned int> h_graph_nodes_start;


    

    //read the start_vertex node from the file
  //set the start_vertex node as 1 in the mask


    std::vector<MyUint1> h_updating_graph_mask(numCols,0);
    // make this a different datatype and cast it to the kernel
    // hpm version of the stratix 10 try 
  std::vector<MyUint1> h_graph_visited(numCols,0); 
h_graph_visited[start_vertex]=1; 


  GraphData fpga_cu_data;


  int numEdges=0;
  // iterate over num of compute units to generate the graph partition data
  for(int indexPE = 0; indexPE < NUM_COMPUTE_UNITS; indexPE++){

    int indptr_end = old_buffer_size_indptr[indexPE+1];

  int inds_end = old_buffer_size_inds[indexPE+1];
  // initalize the memory


  
  	numEdges  += source_meta[2 + old_buffer_size_meta[indexPE]];  // nonZ count -> total edges



    HostGraphDataGenerate(indexPE,start_vertex,fpga_cu_data,source_meta,source_indptr,source_inds,old_buffer_size_meta,old_buffer_size_indptr,old_buffer_size_inds);
  }
  // Matrix Graph;
  // Graph.Populate(start_vertex,numCols,numEdges,source_indptr,source_inds,h_dist,h_updating_graph_mask,h_graph_visited));

  run_bfs_fpga<8>(numCols,
                  source_inds,
                  source_indptr,
                  h_updating_graph_mask,
                  h_graph_visited,
                  h_dist,
                  start_vertex,numEdges);  


 


  
  // unsigned int newarr[16] = {1013, 1083, 1260, 1385, 847, 484, 1351, 1050, 430, 742, 344, 331, 216, 12, 1386, 816};
      // cout << "Time sequential: " << (elapsed_seq/1000000000)  << " -  " <<CPU_bfstimeInSeconds << " sec\n";


  // verify
   // For CPU test run
  
  int cpuCUnum=0;
  // this is different than the kernel rows because we want to use the whole matrix to run on cpu not a partition




  // initalize the memory again
  std::vector<unsigned int> host_nodes_start(numCols);
  std::vector<unsigned int> host_nodes_end(numCols);
  std::vector<unsigned int> host_graph_mask(numCols,0);
  std::vector<unsigned int> host_updating_graph_mask(numCols,0);
  std::vector<unsigned int> host_graph_visited(numCols,0);
  // allocate mem for the result on host side
  std::vector<int> host_level(numCols,-1);
    
  int indptr_start = old_buffer_size_indptr_cpu[cpuCUnum]/NUM_COMPUTE_UNITS;
  int indptr_end = old_buffer_size_indptr_cpu[cpuCUnum]/NUM_COMPUTE_UNITS + old_buffer_size_indptr_cpu[cpuCUnum+1]/NUM_COMPUTE_UNITS;
// std::cout << " __ _ _ _ __ _ indptr ______host_____ \n";
//  DEBUG(indptr_start);
//  DEBUG(indptr_end);
//  DEBUG("");
  // initalize the memory
  for(int i = indptr_start,index=0; i < indptr_end-1; index++,i++){
    host_nodes_start[index] =  source_indptr_cpu[i];
    host_nodes_end[index] = source_indptr_cpu[i+1];
  }

  //set the start_vertex node as 1 in the mask

  host_graph_mask[start_vertex]=1;
  host_graph_visited[start_vertex]=1;
  host_level[start_vertex]=0; 





 
  run_bfs_cpu(numCols,source_indptr_cpu,source_inds_cpu, host_graph_mask, host_updating_graph_mask, host_graph_visited, host_level);

    // Select the element with the maximum value
    auto it = std::max_element(host_level.begin(), host_level.end());
    // Check if iterator is not pointing to the end of vector
    int maxLevelCPU = (*it +2);


  print_levels(host_level,"cpu",h_dist,"fpga",maxLevelCPU); // CPU Results


  return 0;
}