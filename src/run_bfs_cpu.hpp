//----------------------------------------------------------
//--bfs on cpu
//--programmer:  jianbin
//----------------------------------------------------------
//---------------------------------------------------------- CPU COMPUTATION
void run_bfs_cpu(int no_of_nodes,
  std::vector<Uint32> &source_indptr,
  std::vector<Uint32>&source_inds, 
  std::vector<Uint32>&h_graph_mask,
  std::vector<Uint32>&h_updating_graph_mask, 
  std::vector<Uint32>&fpga_visited,
  std::vector<int> &h_cost_ref,
  nlohmann::json &newJsonObj,
  std::vector<Uint32> &h_visit_offsets,
    std::vector<DeviceInfo> &host_run_statistics){
  char stop;
  Uint32 exploredEdgesCount = 0;
  std::vector<int> Edgecounts(NUM_GPU, 0);
  std::vector<int> Nodecounts(NUM_GPU, 0);
  do{
    //if no thread changes this value then the loop stops
    stop=0;
    for(int tid = 0; tid < no_of_nodes; tid++ )
    {
      if (h_graph_mask[tid] == 1){ 
        h_graph_mask[tid]=0;
        exploredEdgesCount += source_indptr[tid+1] - source_indptr[tid];
#if VERBOSE == 1  
      for(int i=source_indptr[tid]; i<(source_indptr[tid+1]); i++){
                int id = source_inds[i];
                for (int j = 0; j < NUM_GPU + 1; ++j) {
                  if (id >= h_visit_offsets[j] && id < h_visit_offsets[j + 1]) {
                    Edgecounts[j] += 1;  // Increment the count for the corresponding range
                    Nodecounts[j] += 1;
                    break;  // Exit the inner loop once the correct range is found
                  }
                }
        }
#endif
        for(int i=source_indptr[tid]; i<(source_indptr[tid+1]); i++){
          // int id = source_inds[i+9140365];  //--h_graph_edges is source_inds
          int id = source_inds[i];  // Single Processing Element--h_graph_edges is source_inds
          if(!fpga_visited[id]){  //--cambine: if node id has not been visited, enter the body below
            h_cost_ref[id]=h_cost_ref[tid]+1;
            h_updating_graph_mask[id]=1;
          }
        }
      }    
    }

    for(int tid=0; tid< no_of_nodes ; tid++ )
    {
      if (h_updating_graph_mask[tid] == 1){
        h_graph_mask[tid]=1;
        fpga_visited[tid]=1;
        stop=1;
        h_updating_graph_mask[tid]=0;
      }
    }
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

