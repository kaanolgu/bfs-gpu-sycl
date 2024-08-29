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



//-------------------------------------------------------------------
//--initialize array with maximum limit
//-------------------------------------------------------------------
template<typename datatypeA, typename datatypeB>
void print_levels(std::vector<datatypeA>& A, std::string nameA, std::vector<datatypeB>& B, std::string nameB, int size) {
    // Define width and separator for formatting
    const int columnWidth = 25;

    
    // Print header
    printSeparator();
    std::cout << "|                    VERIFY RESULTS                 |\n";
    printSeparator();
    std::cout << "| " << std::setw(columnWidth) << std::left << nameA 
              << "| " << std::setw(columnWidth) << std::left << nameB << "|\n";
    printSeparator();
    
    // Print levels and their counts
    for (int i = 0; i < size; i++) {
        int countA = std::count(A.begin(), A.end(), i);
        int countB = std::count(B.begin(), B.end(), i);
        
        std::cout << "| " << std::setw(columnWidth - 2) << std::left 
                  << "Level #" + std::to_string(i) + ": " + std::to_string(countA) 
                  << "| " << std::setw(columnWidth - 2) << std::left 
                  << std::to_string(countB) << "|\n";
    }
    
    // Print footer
    printSeparator();
    
    // Verify results and print the test status
    bool passed = std::equal(A.begin(), A.end(), B.begin());
    
    if (passed) {
        std::cout << "| " << std::setw(52) << std::left << "TEST PASSED!" << "|\n";
    } else {
        std::cout << "| " << std::setw(52) << std::left << "TEST FAILED!" << "|\n";
    }
    
    printSeparator();
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
  
  if(NUM_GPU > 1){
    GPURun(numRows,graph.indsMulti,graph.indptrMulti,h_updating_graph_mask,h_graph_visited,h_dist,start_vertex,num_runs,newJsonObj);  
  }else{
    GPURun(numRows,graph.inds,graph.indptr,h_updating_graph_mask,h_graph_visited,h_dist,start_vertex,num_runs,newJsonObj); 
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

  run_bfs_cpu(numCols,graph_cpu.indptr,graph_cpu.inds, host_graph_mask, host_updating_graph_mask, host_graph_visited, host_level,newJsonObj);

  // Select the element with the maximum value
  auto it = std::max_element(host_level.begin(), host_level.end());
  // Check if iterator is not pointing to the end of vector
  int maxLevelCPU = (*it +2);

  print_levels(host_level,"cpu",h_dist,"fpga",maxLevelCPU); // CPU Results


    newJsonObj["num_gpus"] = NUM_GPU; // Adding an array
    newJsonObj["dataset"] = datasetName;
    newJsonObj["avgMTEPS"] = (static_cast<unsigned int>(newJsonObj["edgesCount"])/(1000000*static_cast<double>(newJsonObj["avgExecutionTime"])*1e-3));
    newJsonObj["avgMTEPSFilter"] = (static_cast<unsigned int>(newJsonObj["edgesCount"])/(1000000*static_cast<double>(newJsonObj["avgExecutionTimeFiltered"])*1e-3));
    newJsonObj["coverage"] = static_cast<double>(newJsonObj["edgesCount"]) / numEdges * 100.0;



    // Variable to hold the combined JSON data
    nlohmann::json combinedJsonArray;

    // Read existing data from the file, if it exists
    std::ifstream inFile("data.json");
    if (inFile.is_open()) {
        try {
            // Parse the existing JSON array from the file
            inFile >> combinedJsonArray;
            // Check if the existing data is an array, if not create a new array
            if (!combinedJsonArray.is_array()) {
                combinedJsonArray = nlohmann::json::array();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading or parsing existing JSON data: " << e.what() << std::endl;
            // Initialize as a new array if reading fails
            combinedJsonArray = nlohmann::json::array();
        }
        inFile.close();
    } else {
        // If the file doesn't exist, start with an empty array
        combinedJsonArray = nlohmann::json::array();
    }

    // Append the new JSON object to the combined JSON array
    combinedJsonArray.push_back(newJsonObj);

    // Save the combined JSON data back to the file
    std::ofstream outFile("data.json");
    if (outFile.is_open()) {
        outFile << combinedJsonArray.dump(4); // Write the JSON data with 4 spaces indentation
        outFile.close();
        std::cout << "JSON data appended to 'data.json'." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }

  return 0;
}