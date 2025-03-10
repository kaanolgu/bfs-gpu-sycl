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
bool print_levels(std::vector<datatypeA>& A, std::string nameA, std::vector<std::vector<datatypeB>>& B, std::string nameB, int size) {
    // Define width and separator for formatting
    const int columnWidth = 25;

    
    // Print header

    std::cout << "- CPU : ";
    // Print levels and their counts
    for (int i = 0; i < size; i++) {
        uint32_t countA = std::count(A.begin(), A.end(), i);
        std::cout << std::to_string(countA) << ", "; 
    }
        std::cout << "\n- GPU : ";
    // Print levels and their counts
    
    for (int i = 0; i < size; i++) {
        uint32_t countB = std::count(B.back().begin(), B.back().end(), i);
        std::cout << std::to_string(countB) << ", "; 
    }
    
    // Verify results and print the test status
    bool status = std::equal(A.begin(), A.end(), B.back().begin()) && areAllRowsEqualToLast(B,size);

    
    return status;
}


int main(int argc, char * argv[])
{
  nlohmann::json newJsonObj;
  CommandLineParser parser;
  // Register the argument with a default value
  parser.addArgument("num_runs", "1");
  parser.addArgument("dataset", "default");
  parser.addArgument("root", "1");
  parser.addArgument("output", "data.json");
  // Parse the command line arguments
  parser.parseArguments(argc, argv);

  // Get the value of the argument after parsing
  int num_runs = std::stoi(parser.getArgument("num_runs"));
  std::string datasetName = parser.getArgument("dataset").c_str();
  int start_vertex = stoi(parser.getArgument("root"));
  std::string output_json_name = parser.getArgument("output").c_str();

  parser.printArguments();
      std::cout << std::setw(20) << std::left << "- num_gpu" << std::setw(20) << NUM_GPU << std::endl;



  

  CSRGraph graph = loadMatrix(NUM_GPU,datasetName);
  uint32_t numRows =0;
  uint64_t numEdges = 0;

  uint32_t numCols = graph.meta32[1]; // cols -> total number of vertices
    for (size_t i = 0; i < graph.meta32.size(); i += 3) {
        numRows += graph.meta32[i + 0];
    }
    for (size_t i = 0; i < graph.meta64.size(); i += 1) {
        numEdges += graph.meta64[i + 0]; // nonZ count -> total edges
    }

      std::cout << std::setw(20) << std::left << "- # vertices" << std::setw(20) << numCols << std::endl;
      std::cout << std::setw(20) << std::left << "- # edges" << std::setw(20) << numEdges << std::endl;
      std::cout << std::setw(20) << std::left << "- dataset" << std::setw(20) << "Load [OK]" << std::endl;
    #if USE_GLOBAL_LOAD_BALANCE == 1
    std::cout << std::setw(20) << std::left << "- load balance" << std::setw(20) << "[GLOBAL]" << std::endl;
    #else
        #if USE_STRIDED_LOCAL_LOAD_BALANCE
        std::cout << std::setw(20) << std::left << "- load balance" << std::setw(20) << "[STRIDED-LOCAL]" << std::endl;    
        #else
        std::cout << std::setw(20) << std::left << "- load balance" << std::setw(20) << "[LOCAL]" << std::endl;
        #endif
    #endif
      std::cout <<"----------------------------------------"<< std::endl;
   // Sanity check if we loaded graph properly
   assert(numRows <= numCols);
 


  // GPU
  std::vector<int> h_dist(numCols,-1);
  h_dist[start_vertex]=0;
  std::vector<std::vector<int>> h_distancesGPU(num_runs,h_dist);
  std::vector<uint32_t> h_graph_nodes_start;
  std::vector<uint8_t> h_updating_graph_mask(numCols,0);
  std::vector<uint8_t> h_graph_visited(numCols,0);

  
  h_graph_visited[start_vertex]=1;
  std::vector<uint32_t>  h_visit_offsets(NUM_GPU+1,0);


    std::vector<uint32_t> selected = {0}; // Start with 0
    // Select elements at indices that are multiples of 4
    for (size_t i = 0; i < graph.meta32.size(); i += 3) {
        selected.push_back(graph.meta32[i]);
    }

    
    // Compute partial sum and store the result, starting from position 1
    std::partial_sum(selected.begin(), selected.end(), h_visit_offsets.begin());


  #if VERBOSE ==1 
for (size_t i = 0; i < graph.meta32.size(); i += 1) {
    std::cout << "graph meta32[" << i <<"]: "<<  graph.meta32[i] << std::endl;
}

for( uint32_t i =0; i < h_visit_offsets.size(); i++)
std::cout << "- GPU[" << std::to_string(i) << "] OFFSET:\t" << h_visit_offsets[i] << std::endl;
std::cout <<"----------------------------------------"<< std::endl;
#endif

  if(NUM_GPU > 1){
    GPURun(numRows,graph.indsMulti,graph.indptrMulti,h_updating_graph_mask,h_graph_visited,h_distancesGPU,start_vertex,num_runs,newJsonObj,h_visit_offsets);  
  }else{
    GPURun(numRows,graph.inds,graph.indptr,h_updating_graph_mask,h_graph_visited,h_distancesGPU,start_vertex,num_runs,newJsonObj,h_visit_offsets); 
  }

  

  
  //////////////////////////////////////////////////
  // CPU
  //////////////////////////////////////////////////


  // initalize the memory again
  std::vector<uint8_t> host_graph_mask(numCols,0);
  std::vector<uint8_t> host_updating_graph_mask(numCols,0);
  std::vector<uint8_t> host_graph_visited(numCols,0);
  std::vector<int> host_level(numCols,-1);
    
  //set the start_vertex node as 1 in the mask
  host_graph_mask[start_vertex]=1;
  host_graph_visited[start_vertex]=1;
  host_level[start_vertex]=0; 
  std::vector<DeviceInfo> host_run_statistics;

  if(NUM_GPU > 1){
    run_bfs_cpu(numRows,graph.indptrMulti,graph.indsMulti, host_graph_mask, host_updating_graph_mask, host_graph_visited, host_level,newJsonObj,h_visit_offsets,host_run_statistics);
  }else{
    run_bfs_cpu(numRows,graph.indptr,graph.inds, host_graph_mask, host_updating_graph_mask, host_graph_visited, host_level,newJsonObj,h_visit_offsets,host_run_statistics);
  }
  // Select the element with the maximum value
  // Use GPU results because in large scales we won't have CPU results to validate
  // We could validate by running local and global models and cross check
  auto it = std::max_element(h_distancesGPU[0].begin(), h_distancesGPU[0].end());
  // Check if iterator is not pointing to the end of vector
  int maxLevelCPU = (*it +2);
    std::cout <<"----------------------------------------"<< std::endl;
    bool status = print_levels(host_level,"cpu",h_distancesGPU,"fpga",maxLevelCPU); // CPU Results
    newJsonObj["valid"] = status;
    std::cout <<"\n----------------------------------------"<< std::endl;
    std::cout << (status ? "[SUCCESS]" : "[FAILURE]") << "\tAll results on GPUs (" << num_runs << " runs) cross checked vs CPU results match\n";
    std::cout <<"----------------------------------------"<< std::endl;

    // newJsonObj["num_gpus"] = NUM_GPU; // Adding an array
    // newJsonObj["dataset"] = datasetName;
    newJsonObj["startVertex"] = start_vertex;
    newJsonObj["numLevels"] = maxLevelCPU - 1;
    newJsonObj["avgMTEPS"] = (static_cast<unsigned int>(newJsonObj["edgesCount"])/(1000000*static_cast<double>(newJsonObj["avgExecutionTime"])*1e-3));
    newJsonObj["avgMTEPS90f"] = (static_cast<unsigned int>(newJsonObj["edgesCount"])/(1000000*static_cast<double>(newJsonObj["avgExecutionTime90f"])*1e-3));
    double gtepsValue = static_cast<unsigned int>(newJsonObj["edgesCount"]) / (1000000000 * static_cast<double>(newJsonObj["avgExecutionTime90f"]) * 1e-3);
    std::ostringstream streamObj;
    streamObj << std::fixed << std::setprecision(2) << gtepsValue;
    newJsonObj["avgGTEPS90f"] = streamObj.str();

    newJsonObj["avgMTEPSFilter"] = (static_cast<unsigned int>(newJsonObj["edgesCount"])/(1000000*static_cast<double>(newJsonObj["avgExecutionTimeFiltered"])*1e-3));
    newJsonObj["maxMTEPSFilter"] = (static_cast<unsigned int>(newJsonObj["edgesCount"])/(1000000*static_cast<double>(newJsonObj["minExecutionTimeFiltered"])*1e-3));
    newJsonObj["edgesCoverage"] = static_cast<uint64_t>(newJsonObj["edgesCount"]) / numEdges * 100.0;
    // Add counts from the second dimension (A.back()) directly into "Levels" key
    for (int i = 0; i < maxLevelCPU - 1; i++) {
        int countA = std::count(h_distancesGPU.back().begin(), h_distancesGPU.back().end(), i);
        newJsonObj["LevelsGPU"].push_back(countA); // Directly append to "Levels"
    }


#if VERBOSE == 1
std::cout << std::endl;
std::cout << std::endl;
std::cout << std::endl;
printDeviceInfo(host_run_statistics);   
std::cout << std::endl;
std::cout << std::endl;
std::cout << std::endl;


#endif






    // Variable to hold the combined JSON data
    nlohmann::json combinedJsonObj;

    // Read existing data from the file, if it exists
    std::ifstream inFile(output_json_name);
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
#if USE_GLOBAL_LOAD_BALANCE == 1
    std::string datasetKey = datasetName + "_" + std::to_string(num_runs) + "_" + "global";
#else
    #if USE_STRIDED_LOCAL_LOAD_BALANCE
        std::string datasetKey = datasetName + "_" + std::to_string(num_runs) + "_" + "local_strided";
    #else
        std::string datasetKey = datasetName + "_" + std::to_string(num_runs) + "_" + "local";
    #endif
#endif
    
    // Add the new JSON object under the key corresponding to NUM_GPU
    combinedJsonObj[datasetKey][std::to_string(NUM_GPU)] = newJsonObj;

    // Write the updated JSON object back to the file
    std::ofstream outFile(output_json_name);
    if (outFile.is_open()) {
        outFile << combinedJsonObj.dump(4); // Pretty-print with 4-space indentation
        outFile.close();
    } else {
        std::cerr << "Error opening file for writing." << std::endl;
    }



  return 0;
}