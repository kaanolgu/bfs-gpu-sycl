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


   std::cout<<"Missing ones in GPU : ";
for(int level=0; level < size; level++){
  
  int sum =0;
int nok_sum =0;
std::cout << "level :" << level << " ";
  for (size_t i = 0; i < B.size(); ++i) {
        if (B[i] == level) {
            
            if (A[i] != level) {
              nok_sum+=1;
            //     std::cout << "ERROR at index " << i << ", A[i] = " << A[i] << ", B[i] = " << B[i] << std::endl;
            }else{
                sum +=1;
            }
        }
        if(B[i] != level){
          if(A[i] == level){
              // std::cout << i << "[" << B[i] << "], ";
          }
        }
    }
    std::cout << "\n";
    std::cout<<"Level "<< level <<" \nMatching -> "<< sum << std::endl;
    std::cout<<"Non Matching -> "<< nok_sum << std::endl;
 
}
}

int main(int argc, char * argv[])
{
  
  CommandLineParser parser;
  // Register the argument with a default value
  parser.addArgument("num_runs", "1");
  parser.addArgument("dataset", "default");
  parser.addArgument("root", "1");
  parser.addArgument("num_gpus", "1");
  // Parse the command line arguments
  parser.parseArguments(argc, argv);

  // Get the value of the argument after parsing
  int num_runs = std::stoi(parser.getArgument("num_runs"));
  std::string datasetName = parser.getArgument("dataset").c_str();
  int start_vertex = stoi(parser.getArgument("root"));
  int num_gpus =  stoi(parser.getArgument("num_gpus"));

  parser.printArguments();



	std::vector<Uint32> old_buffer_size_meta(1,0);
	std::vector<Uint32> old_buffer_size_indptr(1,0);
	std::vector<Uint32> old_buffer_size_inds(1,0);
	std::vector<Uint32> old_buffer_size_config(1,0);
  Uint32 offset_meta =0;
  Uint32 offset_indptr =0;
  Uint32 offset_inds =0;

  
  std::cout << "######################LOADING MATRIX#########################" << std::endl;
  CSRGraph graph = loadMatrix(num_gpus,datasetName);



  CSRGraph graph_cpu = loadMatrix(1,datasetName);
  std::cout << "#############################################################\n" << std::endl;
  numCols = graph_cpu.meta[1];  // cols -> total number of vertices
  std::cout << "number of vertices - initial "<< numCols << std::endl;
  ////////////
  // GPU
  ///////////
  // Dummy example total neighbours needs to be 74
  if(NUM_OF_GPUS > 1){
  for(int i = 0; i < num_gpus; i++){
  std::cout << "i: " << start_vertex<< ", num_Neighbours: "<< (graph.indptrMulti[i][start_vertex+1] - graph.indptrMulti[i][start_vertex]) << std::endl;
  std::cout << "i: " << 135368<< ", num_Neighbours: "<< (graph.indptrMulti[i][135368+1] - graph.indptrMulti[i][135368]) << std::endl;
  }

std::cout << "begin addr : " << graph.indptrMulti[0][135368] << std::endl;
  
  }
  // allocate mem for the result on host side
  std::vector<Uint32> h_dist(numCols,-1);
    // std::cout << "number of vertices "<< numCols << std::endl;

  h_dist[start_vertex]=0; 
  // h_dist[0]=1;  

  std::vector<Uint32> h_graph_nodes_start;
  //read the start_vertex node from the file
  //set the start_vertex node as 1 in the mask
  std::vector<MyUint1> h_updating_graph_mask(numCols,0);
    std::cout << "number of vertices "<< numCols << std::endl;

  // make this a different datatype and cast it to the kernel
  // hpm version of the stratix 10 try 
  std::vector<MyUint1> h_graph_visited(numCols,0); 
    std::cout << "number of vertices "<< numCols << std::endl;

  h_graph_visited[start_vertex]=1;
  // h_graph_visited[0]=1; 

  int indptr_end = old_buffer_size_indptr[1];
  int inds_end = old_buffer_size_inds[1];
  // initalize the memory
  int numEdges  = graph_cpu.meta[2];  // nonZ count -> total edges
  numRows  = graph_cpu.meta[0];  // this it the value we want! (rows)
  // Sanity Check if we loaded the graph properly
  assert(numRows <= numCols);
    std::cout << "number of vertices "<< numCols << std::endl;

  std::cout << std::setw(6) << std::left << "# Graph Information" << "\n Vertices (nodes) = " << numRows << " \n Edges = "<< numEdges << "\n";
  std::cout << "number of vertices "<< numCols << std::endl;
  if(NUM_OF_GPUS > 1){
  GPURun(numRows,
                  graph.indsMulti,
                  graph.indptrMulti,
                  h_updating_graph_mask,
                  h_graph_visited,
                  h_dist,
                  start_vertex,numEdges,
                  num_runs);  
  }else{
    GPURun(numRows,
                  graph.inds,
                  graph.indptr,
                  h_updating_graph_mask,
                  h_graph_visited,
                  h_dist,
                  start_vertex,numEdges,
                  num_runs); 
  }

  

  
  //////////////////////////////////////////////////
  // CPU
  //////////////////////////////////////////////////


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

  run_bfs_cpu(numCols,graph_cpu.indptr,graph_cpu.inds, host_graph_mask, host_updating_graph_mask, host_graph_visited, host_level);

  // Select the element with the maximum value
  auto it = std::max_element(host_level.begin(), host_level.end());
  // Check if iterator is not pointing to the end of vector
  int maxLevelCPU = (*it +2);

  print_levels(host_level,"cpu",h_dist,"fpga",maxLevelCPU); // CPU Results


  return 0;
}