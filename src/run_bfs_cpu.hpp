//----------------------------------------------------------
//--bfs on cpu with multi-dimensional indptr and indices
//--programmer: jianbin (modified)
//----------------------------------------------------------
template<typename vectorT>
void run_bfs_cpu(uint32_t no_of_nodes,
  vectorT &source_indptr,  // Multi-dimensional source_indptr
  vectorT &source_inds,    // Multi-dimensional source_inds
  std::vector<uint8_t> &h_graph_mask,
  std::vector<uint8_t> &h_updating_graph_mask,
  std::vector<uint8_t> &fpga_visited,
  std::vector<int> &h_cost_ref,
  nlohmann::json &newJsonObj,
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
          for (uint32_t i = source_indptr[tid]; i < source_indptr[tid+ 1]; i++) {
            uint32_t id = source_inds[i];
            if (!fpga_visited[id]) {  // if node id has not been visited
              h_cost_ref[id] = level + 1;
              h_updating_graph_mask[id] = 1;
              fpga_visited[id] = 1;
            }
          }
      } else if constexpr (std::is_same_v<vectorT, std::vector<std::vector<uint32_t>>>) {
          for (int j = 0; j < NUM_GPU; ++j) {
            
          exploredEdgesCount += source_indptr[j][tid + 1] - source_indptr[j][tid];

          // Process edges for the current node
          for (uint32_t i = source_indptr[j][tid]; i < source_indptr[j][tid+ 1]; i++) {
            uint32_t id = source_inds[j][i];
            if (!fpga_visited[id]) {  // if node id has not been visited
              h_cost_ref[id] = level + 1;
              h_updating_graph_mask[id] = 1;
              fpga_visited[id] = 1;
            }
          }
        }
      } // end: else if constexpr
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
    DeviceInfo new_info = {i, Edgecounts[i], (double)(Edgecounts[i] / std::accumulate(Edgecounts.begin(), Edgecounts.end(), 0.0)) * 100};
    host_run_statistics.push_back(new_info);
  }
  #endif
  
  newJsonObj["edgesCount"] = exploredEdgesCount;
}
