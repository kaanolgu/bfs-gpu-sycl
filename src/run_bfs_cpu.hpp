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
  std::vector<int> &h_cost_ref){
  char stop;
  do{
    //if no thread changes this value then the loop stops
    stop=0;
    for(int tid = 0; tid < no_of_nodes; tid++ )
    {
      if (h_graph_mask[tid] == 1){ 
        h_graph_mask[tid]=0;
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
}



/////////////////////// ALTERNATIVE UNOPTIMISED VERSION

//  auto graph_cpu = [source,&m_resultCount,&m_nextDistance] (std::vector<unsigned int>&sw_bfs_results,
// 		std::vector<unsigned int>&source_indptr,
// 		std::vector<unsigned int>&source_inds) -> double{


//   std::vector<unsigned int> m_bitmap((numVertices),0);
//   std::vector<int> queue;
//   unsigned int neighbors_start,neighbors_end, neighbor;
  
//   sw_bfs_results[source] = 0;
//   setBit(source,m_bitmap);
//   queue.push_back(source);
  
//   // Timer Start
//   double elapsed = 0;
//   // dpc_common::TimeInterval timer;

//   while (!queue.empty())
//   {
//     int currentVertex = queue.front();
//     m_nextDistance = sw_bfs_results[currentVertex] + 1;

//     queue.erase(queue.begin());
//     neighbors_start = source_indptr[currentVertex];
//     neighbors_end = source_indptr[currentVertex + 1];

// 	for (unsigned int e = neighbors_start; e < neighbors_end; e++) {
// 		neighbor = source_inds[e];
// 		if (testBit(neighbor,m_bitmap) == false)
// 		{
// 			setBit(neighbor,m_bitmap);
// 			queue.push_back(neighbor);
// 			m_resultCount++;
// 			sw_bfs_results[neighbor] = m_nextDistance;
// 		}
//     }

//   }
//   // Timer End
//   // elapsed += timer.Elapsed();

//   return elapsed;
//   };