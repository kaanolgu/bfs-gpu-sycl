//----------------------------------------------------------
//--bfs on cpu
//--programmer:  jianbin
//----------------------------------------------------------
//---------------------------------------------------------- CPU COMPUTATION
void run_bfs_cpu(int no_of_nodes,
  std::vector<unsigned int> &source_indptr,
  std::vector<unsigned int>&source_inds, 
  std::vector<unsigned int>&h_graph_mask,
  std::vector<unsigned int>&h_updating_graph_mask, 
  std::vector<unsigned int>&fpga_visited,
  std::vector<int> &h_cost_ref,
  nlohmann::json &newJsonObj){
  char stop;
  unsigned int exploredEdgesCount = 0;
  do{
    //if no thread changes this value then the loop stops
    stop=0;
    for(int tid = 0; tid < no_of_nodes; tid++ )
    {
      if (h_graph_mask[tid] == 1){ 
        h_graph_mask[tid]=0;
        exploredEdgesCount += source_indptr[tid+1] - source_indptr[tid];
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

  newJsonObj["edgesCount"] = exploredEdgesCount;
}

