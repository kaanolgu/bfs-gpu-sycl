//----------------------------------------------------------
//--bfs on cpu
//--programmer:  jianbin
//----------------------------------------------------------
//---------------------------------------------------------- CPU COMPUTATION
void run_bfs_cpu(uint32_t no_of_nodes,
  std::vector<uint32_t> &source_indptr,
  std::vector<uint32_t>&source_inds, 
  std::vector<uint8_t>&h_graph_mask,
  std::vector<uint8_t>&h_updating_graph_mask, 
  std::vector<uint8_t>&fpga_visited,
  std::vector<int> &h_cost_ref,
  nlohmann::json &newJsonObj,
  std::vector<uint32_t> &h_visit_offsets,
    std::vector<DeviceInfo> &host_run_statistics){
  char stop;
  uint32_t exploredEdgesCount = 0;
  #if VERBOSE == 1  
  std::vector<uint32_t> Edgecounts(NUM_GPU, 0);
  #endif
  int level =0;
  do{
    //if no thread changes this value then the loop stops
    stop=0;
    for(uint32_t tid = 0; tid < no_of_nodes; tid++ )
    {
      if (h_graph_mask[tid] == 1){ 
        h_graph_mask[tid]=0;
        exploredEdgesCount += source_indptr[tid+1] - source_indptr[tid];
#if VERBOSE == 1  
      for(uint32_t i=source_indptr[tid]; i<(source_indptr[tid+1]); i++){
                uint32_t id = source_inds[i];
                for (int j = 0; j < NUM_GPU + 1; ++j) {
                  if (id >= h_visit_offsets[j] && id < h_visit_offsets[j + 1]) {
                    Edgecounts[j] += 1;  // Increment the count for the corresponding range

                    break;  // Exit the inner loop once the correct range is found
                  }
                }
        }
#endif
// std::cout << source_indptr[tid+1] << " ** " << source_indptr[tid] << std::endl;
        for(uint32_t i=source_indptr[tid]; i<(source_indptr[tid+1]); i++){
          // int id = source_inds[i+9140365];  //--h_graph_edges is source_inds
          uint32_t id = source_inds[i];  // Single Processing Element--h_graph_edges is source_inds
          if(!fpga_visited[id]){  //--cambine: if node id has not been visited, enter the body below
            h_cost_ref[id]=level+1;
            h_updating_graph_mask[id]=1;
            fpga_visited[id]=1;
          }
        }
      }    
    }


    for(uint32_t tid=0; tid< no_of_nodes ; tid++ )
    {
      if (h_updating_graph_mask[tid] == 1){
        h_graph_mask[tid]=1;
        stop=1;
        h_updating_graph_mask[tid]=0;
      }
    }
    level++;
  }
  while(stop);

#if VERBOSE == 1  
for(int i =0; i < Edgecounts.size(); i++){
  DeviceInfo new_info = {i, Edgecounts[i], (double)(Edgecounts[i] /  std::accumulate(Edgecounts.begin(), Edgecounts.end(), 0.0)) * 100};
  host_run_statistics.push_back(new_info);
}
#endif
  newJsonObj["edgesCount"] = exploredEdgesCount;
}

