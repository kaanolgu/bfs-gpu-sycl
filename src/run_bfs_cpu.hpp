#include <H5Cpp.h>
void saveCpuReference(const std::string &h5path,
                      const std::vector<int> &host_level,
                      int start_vertex) {
    H5::H5File file(h5path, H5F_ACC_RDWR);
    
    // Create cpu_reference group if it doesn't exist
    H5::Group refGroup;
    if (H5Lexists(file.getId(), "cpu_reference", H5P_DEFAULT) > 0) {
        refGroup = file.openGroup("cpu_reference");
    } else {
        refGroup = file.createGroup("cpu_reference");
    }
    
    std::string dsName = "root_" + std::to_string(start_vertex);
    
    // Delete if exists (re-run with same root)
    if (H5Lexists(refGroup.getId(), dsName.c_str(), H5P_DEFAULT) > 0) {
        refGroup.unlink(dsName);
    }
    
    hsize_t dims = host_level.size();
    H5::DataSpace space(1, &dims);
    H5::DataSet ds = refGroup.createDataSet(dsName, H5::PredType::NATIVE_INT32, space);
    ds.write(host_level.data(), H5::PredType::NATIVE_INT32);
}
bool loadCpuReference(const std::string &h5path,
                      std::vector<int> &host_level,
                      int start_vertex) {
    H5::H5File file(h5path, H5F_ACC_RDONLY);
    
    if (H5Lexists(file.getId(), "cpu_reference", H5P_DEFAULT) <= 0) {
        return false;
    }
    
    std::string dsName = "root_" + std::to_string(start_vertex);
    std::string fullPath = "cpu_reference/" + dsName;
    
    if (H5Lexists(file.getId(), fullPath.c_str(), H5P_DEFAULT) <= 0) {
        return false;
    }
    
    H5::Group grp = file.openGroup("cpu_reference");
    H5::DataSet ds = grp.openDataSet(dsName);
    hsize_t n = ds.getSpace().getSimpleExtentNpoints();
    host_level.resize(n);
    ds.read(host_level.data(), H5::PredType::NATIVE_INT32);
    return true;
}

//----------------------------------------------------------
//--bfs on cpu with multi-dimensional indptr and indices
//--programmer: jianbin (modified)
//----------------------------------------------------------
#include "functions.hpp"
template <typename vectorT>
void run_bfs_cpu(uint32_t no_of_nodes,
                 vectorT &source_indptr,  // Multi-dimensional source_indptr
                 vectorT &source_inds,    // Multi-dimensional source_inds
                 std::vector<uint8_t> &h_graph_mask,
                 std::vector<uint8_t> &h_updating_graph_mask,
                 std::vector<uint8_t> &fpga_visited,
                 std::vector<int> &h_cost_ref, nlohmann::json &newJsonObj,
                 std::vector<uint32_t> &h_visit_offsets,
                 std::vector<DeviceInfo> &host_run_statistics) {
  char stop;
  uint64_t exploredEdgesCount = 0;

#if VERBOSE == 1
  std::vector<uint32_t> Edgecounts(NUM_GPU, 0);
#endif

  int level = 0;
  do {
    // if no thread changes this value, then the loop stops
    stop = 0;

    // Iterate over each partition based on h_visit_offsets

    for (uint32_t tid = 0; tid < no_of_nodes; tid++) {
      if (h_graph_mask[tid] == 1) {
        h_graph_mask[tid] = 0;
        if constexpr (std::is_same_v<vectorT, std::vector<uint32_t>>) {
          exploredEdgesCount += source_indptr[tid + 1] - source_indptr[tid];

          // Process edges for the current node
          for (uint32_t i = source_indptr[tid]; i < source_indptr[tid + 1];
               i++) {
            uint32_t id = source_inds[i];
            if (!fpga_visited[id]) {  // if node id has not been visited
              h_cost_ref[id] = level + 1;
              h_updating_graph_mask[id] = 1;
              fpga_visited[id] = 1;
            }
          }
        } else if constexpr (std::is_same_v<
                                 vectorT, std::vector<std::vector<uint32_t>>>) {
          for (int j = 0; j < NUM_GPU; ++j) {
            exploredEdgesCount +=
                source_indptr[j][tid + 1] - source_indptr[j][tid];

            // Process edges for the current node
            for (uint32_t i = source_indptr[j][tid];
                 i < source_indptr[j][tid + 1]; i++) {
              uint32_t id = source_inds[j][i];
              if (!fpga_visited[id]) {  // if node id has not been visited
                h_cost_ref[id] = level + 1;
                h_updating_graph_mask[id] = 1;
                fpga_visited[id] = 1;
              }
            }
          }
        }  // end: else if constexpr
      }
    }

    // Update the graph mask for the next level
    for (uint32_t tid = 0; tid < no_of_nodes; tid++) {
      if (h_updating_graph_mask[tid] == 1) {
        h_graph_mask[tid] = 1;
        stop = 1;
        h_updating_graph_mask[tid] = 0;
      }
    }
    level++;
  } while (stop);

#if VERBOSE == 1
  for (int i = 0; i < Edgecounts.size(); i++) {
    DeviceInfo new_info = {
        i, Edgecounts[i],
        (double)(Edgecounts[i] /
                 std::accumulate(Edgecounts.begin(), Edgecounts.end(), 0.0)) *
            100};
    host_run_statistics.push_back(new_info);
  }
#endif

  newJsonObj["edgesCount"] = exploredEdgesCount;
}
