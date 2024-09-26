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
using json = nlohmann::json;


//-------------------------------------------------------------------
//--initialize array with maximum limit
//-------------------------------------------------------------------
template<typename datatypeA, typename datatypeB>
void print_levels(std::vector<datatypeA>& A, std::string nameA, std::vector<datatypeB>& B, std::string nameB, int size) {
    // Define width and separator for formatting
    const int columnWidth = 25;

    
    // Print header
    printSeparator();
    std::cout << "CPU : ";
    // Print levels and their counts
    for (int i = 0; i < size; i++) {
        int countA = std::count(A.begin(), A.end(), i);
        std::cout << std::to_string(countA) << ", "; 
    }
        std::cout << "\nGPU : ";
    // Print levels and their counts
    for (int i = 0; i < size; i++) {
        int countB = std::count(B.begin(), B.end(), i);
        std::cout << std::to_string(countB) << ", "; 
    }
    
    // Verify results and print the test status
    bool passed = std::equal(A.begin(), A.end(), B.begin());
    
    if (passed) {
        std::cout << "\n TEST PASSED!" << "\n\n";
    } else {
        std::cout << "\n TEST FAILED!" << "\n\n";
    }
    

}


int main(int argc, char * argv[])
{
  nlohmann::json newJsonObj;
  CommandLineParser parser;
  // Register the argument with a default value
  parser.addArgument("num_runs", "1");
  parser.addArgument("dataset", "default");
  parser.addArgument("root", "1");
  // Parse the command line arguments
  parser.parseArguments(argc, argv);

  // Get the value of the argument after parsing
  int num_runs = std::stoi(parser.getArgument("num_runs"));
  std::string datasetName = parser.getArgument("dataset").c_str();
  int start_vertex = stoi(parser.getArgument("root"));


  parser.printArguments();



	std::vector<Uint32> old_buffer_size_meta(1,0);
	std::vector<Uint32> old_buffer_size_config(1,0);
  Uint32 offset_meta =0;
  Uint32 offset_indptr =0;
  Uint32 offset_inds =0;

  
  std::cout << "######################LOADING MATRIX#########################" << std::endl;
  CSRGraph graph = loadMatrix(NUM_GPU,datasetName);
  CSRGraph graph_cpu = loadMatrix(1,datasetName);
  std::cout << "#############################################################\n" << std::endl;
  int numCols = graph_cpu.meta[1];  // cols -> total number of vertices
  int numEdges  = graph_cpu.meta[2];  // nonZ count -> total edges
  int numRows   = graph_cpu.meta[0];  // this it the value we want! (rows)
  std::cout << std::setw(6) << std::left << "# Graph Information" << "\n Vertices (nodes) = " << numRows << " \n Edges = "<< numEdges << "\n";

  // Sanity Check if we loaded the graph properly
  assert(numRows <= numCols);


  // GPU
  std::vector<Uint32> h_dist(numCols,-1);
  std::vector<Uint32> h_graph_nodes_start;
  std::vector<MyUint1> h_updating_graph_mask(numCols,0);
  std::vector<MyUint1> h_graph_visited(numCols,0); 

  h_dist[start_vertex]=0; 
  h_graph_visited[start_vertex]=1;
   std::vector<Uint32>  h_visit_offsets(NUM_GPU+1,0);


    std::vector<int> selected = {0}; // Start with 0
    // Select elements at indices that are multiples of 4
    for (size_t i = 0; i < graph.meta.size(); i += 4) {
        selected.push_back(graph.meta[i]);
    }

    
    // Compute partial sum and store the result, starting from position 1
    std::partial_sum(selected.begin(), selected.end(), h_visit_offsets.begin());


  

  if(NUM_GPU > 1){
    GPURun(numRows,graph.indsMulti,graph.indptrMulti,h_updating_graph_mask,h_graph_visited,h_dist,start_vertex,num_runs,newJsonObj,h_visit_offsets);  
  }else{
    GPURun(numRows,graph.inds,graph.indptr,h_updating_graph_mask,h_graph_visited,h_dist,start_vertex,num_runs,newJsonObj,h_visit_offsets); 
  }

  

  
  //////////////////////////////////////////////////
  // CPU
  //////////////////////////////////////////////////


  // initalize the memory again
  std::vector<Uint32> host_graph_mask(numCols,0);
  std::vector<Uint32> host_updating_graph_mask(numCols,0);
  std::vector<Uint32> host_graph_visited(numCols,0);
  std::vector<int> host_level(numCols,-1);
    
  //set the start_vertex node as 1 in the mask
  host_graph_mask[start_vertex]=1;
  host_graph_visited[start_vertex]=1;
  host_level[start_vertex]=0; 

  run_bfs_cpu(numCols,graph_cpu.indptr,graph_cpu.inds, host_graph_mask, host_updating_graph_mask, host_graph_visited, host_level,newJsonObj,h_visit_offsets);

  // Select the element with the maximum value
  auto it = std::max_element(host_level.begin(), host_level.end());
  // Check if iterator is not pointing to the end of vector
  int maxLevelCPU = (*it +2);

  print_levels(host_level,"cpu",h_dist,"fpga",maxLevelCPU); // CPU Results


    // newJsonObj["num_gpus"] = NUM_GPU; // Adding an array
    // newJsonObj["dataset"] = datasetName;
    newJsonObj["avgMTEPS"] = (static_cast<unsigned int>(newJsonObj["edgesCount"])/(1000000*static_cast<double>(newJsonObj["avgExecutionTime"])*1e-3));
    newJsonObj["avgMTEPSFilter"] = (static_cast<unsigned int>(newJsonObj["edgesCount"])/(1000000*static_cast<double>(newJsonObj["avgExecutionTimeFiltered"])*1e-3));
    newJsonObj["maxMTEPSFilter"] = (static_cast<unsigned int>(newJsonObj["edgesCount"])/(1000000*static_cast<double>(newJsonObj["minExecutionTimeFiltered"])*1e-3));
    newJsonObj["edgesCoverage"] = static_cast<double>(newJsonObj["edgesCount"]) / numEdges * 100.0;



    // Variable to hold the combined JSON data
    nlohmann::json combinedJsonObj;

    // Read existing data from the file, if it exists
    std::ifstream inFile("data.json");
    if (inFile.is_open()) {
        try {
            // Parse the existing JSON object from the file
            inFile >> combinedJsonObj;
            // Check if the existing data is an object, if not create a new object
            if (!combinedJsonObj.is_object()) {
                combinedJsonObj = json::object();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading or parsing existing JSON data: " << e.what() << std::endl;
            // Initialize as a new object if reading fails
            combinedJsonObj = json::object();
        }
        inFile.close();
    } else {
        // If the file doesn't exist, start with an empty object
        combinedJsonObj = json::object();
    }
    std::string datasetKey = datasetName + "_" + std::to_string(num_runs);
    // Add the new JSON object under the key corresponding to NUM_GPU
    combinedJsonObj[datasetKey][std::to_string(NUM_GPU)] = newJsonObj;

    // Write the updated JSON object back to the file
    std::ofstream outFile("data.json");
    if (outFile.is_open()) {
        outFile << combinedJsonObj.dump(4); // Pretty-print with 4-space indentation
        outFile.close();
    } else {
        std::cerr << "Error opening file for writing." << std::endl;
    }



  return 0;
}