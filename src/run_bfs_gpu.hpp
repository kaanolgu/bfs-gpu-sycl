using namespace sycl;
 using namespace std::chrono;
// #include <oneapi/dpl/utility> // to get std:: libraries working here: 
// https://oneapi-src.github.io/oneDPL/api_for_sycl_kernels/tested_standard_cpp_api.html
#include <math.h>
#include <iostream>
#include <vector>
// #include <sycl/ext/intel/fpga_extensions.hpp>
#include <bitset>
#include "functions.hpp"
#include "unrolled_loop.hpp"
#define MAX_NUM_LEVELS 100
// TODO: replace with different factors per GPU type, H100 SXM5 SM_CNT 132
#define SM_FACTOR 96
#define SM_CNT 108


// |  Memory Model Equivalence
// |  CUDA                 SYCL
// +------------------------------------+
// |  register        = private memory  |
// |  shared memory   = local memory    |
// |  constant memory = constant memory |
// |  global memory   = global memory   |
// |  local memory    = N/A             |
// +------------------------------------+


// |  Execution Model Equivalence
// |  CUDA          SYCL
// +------------------------+
// |  SM        = CU        |
// |  SM core   = PE        |
// |  thread    = work-item |
// |  block     = work-group|
// +------------------------+


#include <sycl/sycl.hpp>
// This function returns a vector of two (not necessarily distinct) devices,
// allowing computation to be split across said devices.

void printExecutionTimes(const std::vector<double>& execution_timesA, const std::vector<double>& execution_timesB) {
    const int NUM_TIMES = execution_timesA.size();
    const int columnWidth = 9;      // Width for time values
    const int percentageWidth = 5;   // Width for percentage values

    // Calculate grand total
    double grandTotal = std::accumulate(execution_timesA.begin(), execution_timesA.end(), 0.0)
                      + std::accumulate(execution_timesB.begin(), execution_timesB.end(), 0.0);

    // Print header
    std::cout << std::left << std::setw(10) << "GPU ID"
              << std::setw(columnWidth) << "EN [ms]"
              << std::setw(percentageWidth) << "(%)"
              << std::setw(columnWidth) << "LG [ms]"
              << std::setw(percentageWidth) << "(%)"
              << std::setw(columnWidth) << "Total"
              << std::setw(percentageWidth) << "(%)" << std::endl;

    std::cout << std::string(10 + columnWidth * 3 + percentageWidth * 3, '-') << std::endl; // Separator line

    // Print each row
    for (int i = 0; i < NUM_TIMES; ++i) {
        double rowTotal = execution_timesA[i] + execution_timesB[i];
        double percentageA = (execution_timesA[i] / grandTotal) * 100;
        double percentageB = (execution_timesB[i] / grandTotal) * 100;
        double percentageRow = (rowTotal / grandTotal) * 100;

        std::cout << std::left << std::setw(10) << (i + 1)
                  << std::setw(columnWidth) << std::fixed << std::setprecision(3) << execution_timesA[i]
                  << std::setw(percentageWidth) << std::setprecision(2) <<percentageA << "%"
                  << std::setw(columnWidth) << std::fixed << std::setprecision(3) << execution_timesB[i]
                  << std::setw(percentageWidth) << std::setprecision(2) << percentageB << "%"
                  << std::setw(columnWidth) << std::fixed << std::setprecision(3) << rowTotal
                  << std::setw(percentageWidth) << std::setprecision(2) << percentageRow << "%" << std::endl;
    }

    std::cout << std::string(10 + columnWidth * 3 + percentageWidth * 3, '-') << std::endl; // Ending separator line

    // Add notes at the bottom
    std::cout << "* EN: Explore Neighbours Kernel" << std::endl;
    std::cout << "* LG: Levelgen" << std::endl;
}




// Function to find the first and last non-zero indices in a vector, ignoring the specified start index
template <typename T>
std::pair<std::optional<size_t>, std::optional<size_t>> find_first_last_nonzero(
    const std::vector<T>& arr, size_t ignore_index = std::numeric_limits<size_t>::max()) {
    std::optional<size_t> first_nonzero;
    std::optional<size_t> last_nonzero;

    // Traverse the array to find the first and last non-zero indices
    for (size_t i = 0; i < arr.size(); ++i) {
        // Skip the specified index to be ignored
        if (i == ignore_index) {
            continue;
        }

        // Check if the current element is non-zero
        if (arr[i] != 0) {
            if (!first_nonzero.has_value()) {
                first_nonzero = i;  // Set the first non-zero index
            }
            last_nonzero = i;  // Continuously update the last non-zero index
        }
    }

    return {first_nonzero, last_nonzero};
}
// initialize device arr with val, if needed set arr[pos] = pos_val
template <typename T>
void copyToDevice(queue &Q, std::vector<T> &arr, T *usm_arr, size_t offset = 0) {
    // Check if the offset is within the bounds of the array
    if (offset > arr.size()) {
        throw std::out_of_range("Offset is out of the bounds of the array.");
    }

    // Copy data from arr starting from arr.data() + offset
    Q.memcpy(usm_arr, arr.data() + offset, (arr.size() - offset) * sizeof(T)).wait();
}
// initialize device arr with val, if needed set arr[pos] = pos_val
template <typename T>
void copyToHost(queue &Q, T *usm_arr,std::vector<T> &arr){
  Q.memcpy(arr.data(),usm_arr, arr.size() * sizeof(T)).wait();
}

//-------------------------------------------------------------------
// Return the execution time of the event, in seconds
//-------------------------------------------------------------------


double GetExecutionTime(const event &e) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double end_k = e.get_profiling_info<info::event_profiling::command_end>();
  double kernel_time = (end_k - start_k) * 1e-6; // ns to ms
  return kernel_time;
}



//-------------------------------------------------------------------
//-- initialize Kernel for Exploring the neighbours of next to visit 
//-- nodes
//-------------------------------------------------------------------
template <int krnl_id>
class ExploreNeighbours1;
template <int krnl_id>
class ExploreNeighbours2;
template <int krnl_id>
class ExploreNeighbours3;
template <int krnl_id>
class ExploreNeighbours4;
template <int krnl_id>
class ExploreNeighbours5;
template <int krnl_id>
class LevelGen;

template <int krnl_id>
event parallel_explorer_kernel(queue &q,
                                const uint32_t V,
                                uint32_t* usm_nodes_start,
                                uint32_t *usm_edges,
                                uint32_t* usm_pipe_1,
                                uint32_t* usm_local_indptr,
                                uint8_t *usm_visit_mask,
                                uint8_t *usm_visit,
                                uint32_t *usm_scan_temp,
                                uint32_t *usm_scan_full,
                                const uint32_t Vstart,
                                const uint32_t Vsize)
    {
      /*
      uint32_t* debug   = malloc_device<uint32_t>(128, q);
      */

      // Define the work-group size and the number of work-groups
      const size_t local_size = LOCAL_WORK_SIZE;  // Number of work-items per work-group
      const size_t global_size1 = ((V + local_size - 1) / local_size) * local_size;
      const size_t global_size4 = SM_CNT * SM_FACTOR * 1024;

      // Setup the range
      nd_range<1> range1(global_size1, local_size);
      nd_range<1> range4(global_size4, local_size);

      auto e1 = q.submit([&](handler& h) {
        h.parallel_for<class ExploreNeighbours1<krnl_id>>(range1, [=](nd_item<1> item) {
          const uint32_t gid = item.get_global_id(0);        // global id
          const uint32_t lid = item.get_local_id(0);         // threadIdx.x
          const uint32_t blockDim = item.get_local_range(0); // blockDim.x

          uint32_t local_node_deg;

          // 1. Read from usm_pipe, replace entry with corresponding edge pointer
          if (gid < V) {
            uint32_t v = usm_pipe_1[gid];
            usm_local_indptr[gid] = usm_nodes_start[v]; // Replacing pipe entry with edge pointer
            local_node_deg = usm_nodes_start[v+1] - usm_nodes_start[v]; // Calculate each nodes partition specific degree
          } else {
            local_node_deg = 0;
          }
          sycl::group_barrier(item.get_group());

          /* Previous variant using reduce and scan; same behavior
          // 2. Cumulative sum of total number of nonzeros 
          uint32_t total_nnz = reduce_over_group(item.get_group(), local_node_deg, sycl::plus<>());
          sycl::group_barrier(item.get_group());

          // 2.a. Store cummulative sum
          if (lid == 0 && gid < V){ // only write for active edges
            usm_scan_temp[item.get_group(0)] = total_nnz;
          }

          // 3. Locally calculate inclusive sum, but store to global array shifted 
          // by one to obtain exclusive sum of degrees to find total work items per block.
          usm_scan_full[gid+1] = sycl::inclusive_scan_over_group(item.get_group(), local_node_deg, sycl::plus<>());
          */

          // New variant using last result from scan also for block sum
          uint32_t running_sum = sycl::inclusive_scan_over_group(item.get_group(), local_node_deg, sycl::plus<>());
          sycl::group_barrier(item.get_group());
          if(gid < V+blockDim){
            // 2. Outputs _excluxive_ group scan by shifting outputs by 1
            usm_scan_full[gid+1] = running_sum;
            // 3. Outputs block sum
            if (lid == blockDim-1){
              usm_scan_temp[item.get_group(0)] = running_sum;
            }
          }
      });
      });
      #if VERBOSE
      q.wait();
      std::vector<uint32_t> ScanHost(128);
      copyToHost(q,usm_scan_temp,ScanHost);
      std::cout << "ScanTempStep1 ";
      for(int i=0; i<128; i++){
        std::cout << std::setw(10) << ScanHost[i];
      }
      std::cout << std::endl;
      copyToHost(q,usm_scan_full,ScanHost);
      std::cout << "ScanFullStep1 ";
      for(int i=0; i<128; i++){
        std::cout << std::setw(10) << ScanHost[i];
      }
      std::cout << std::endl;
      #endif

      q.submit([&](handler& h) {
      h.parallel_for<class ExploreNeighbours2<krnl_id>>(range1, [=](nd_item<1> item) {
          const uint32_t blockDim  = item.get_local_range(0); // blockDim.x
          // single work group to calculate exclusive scan of block sum over all blocks
          // TODO: might want to spawn with larger work group
          // note: last pointer to algorithm can be out of bounds here, will not be accessed
          if(item.get_group(0) == 0){
            sycl::joint_exclusive_scan(item.get_group(), 
                                        usm_scan_temp, 
                                        usm_scan_temp+(V+blockDim-1)/blockDim+1, 
                                        usm_scan_temp, 
                                        sycl::plus<>());
          }
      });
      });
      #if VERBOSE
      q.wait();
      copyToHost(q,usm_scan_temp,ScanHost);
      std::cout << "ScanTempStep2 ";
      for(int i=0; i<128; i++){
        std::cout << std::setw(10) << ScanHost[i];
      }
      std::cout << std::endl;
      #endif

      q.submit([&](handler& h) {
      h.parallel_for<class ExploreNeighbours3<krnl_id>>(range1, [=](nd_item<1> item) {
          const uint32_t gid       = item.get_global_id(0);    // global id
          const uint32_t blockDim  = item.get_local_range(0);  // blockDim.x
          // const uint32_t total_full = usm_scan_temp[(V+blockDim-1)/blockDim];

          // 5. create global _excluxive_ scan result by adding 
          // previously local result from scan_full and block offset
          if(gid < V){
            usm_scan_full[gid+1] = usm_scan_full[gid+1] + usm_scan_temp[item.get_group(0)];
          } /* else if(gid == V) {
            usm_scan_full[0] = 0; 
          } else {     
            usm_scan_full[gid] = total_full;
          }*/
      });
      });
      #if VERBOSE
      q.wait();
      copyToHost(q,usm_scan_full,ScanHost);      
      std::cout << "ScanFullStep3 ";
      for(int i=0; i<128; i++){
        std::cout << std::setw(10) << ScanHost[i];
      }
      std::cout << std::endl;
      #endif

      q.submit([&](handler& h) {
    h.parallel_for<class ExploreNeighbours4<krnl_id>>(range4, [=](nd_item<1> item) {
            const uint32_t gid = item.get_global_id(0);    // global id
            const uint32_t lid  = item.get_local_id(0); // threadIdx.x
            //const uint32_t blockDim  = item.get_local_range(0); // blockDim.x
            //const uint32_t bid = item.get_group(0); // block id
            //const uint32_t grange = item.get_group_range(0);

    const uint32_t total_full = usm_scan_full[V];
    // 4. Using grid stride loop with binary search to find the corresponding edges 
    for (int i = gid;       // global idx
        i < total_full;     // total degree to process
        i += global_size4   // increment by global_size
    ) {

      // uint32_t it = upper_bound(usm_scan_full, total_full, i);
      // uint32_t start = std::upper_bound(usm_scan_full, usm_scan_full+V, i) - usm_scan_full;
      // manually inlined
      uint32_t start = 0;
      uint32_t temp_length = V;
      int ii;
      while (start < temp_length) {
          ii = start + (temp_length - start) / 2;
          if (i < usm_scan_full[ii]) { 
              temp_length = ii; //!!! add -1?
          } else {
              start = ii + 1; //!!! remove +1?
          }
      } // end while
      // debug output
      //if(i < V)
      //  usm_scan_temp[i] = start;
      
      uint32_t id = start - 1; // !!! remove -1 to use inclusive sum?
      uint32_t  e = usm_local_indptr[id] + i  - usm_scan_full[id];
      uint32_t  n = usm_edges[e];   
      if(!usm_visit[n- Vstart]){
        usm_visit_mask[n - Vstart] = 1;
        usm_visit[n- Vstart] = 1;
      }
      /*
      if(i < 128){
        debug[i] = e;
        usm_scan_temp[i] = n;
      }*/

    } 
      });
          });
      #if VERBOSE
      q.wait();
      /*
      copyToHost(q,debug,ScanHost);
      std::cout << "EdgeID  Step4 ";
      for(int i=0; i<128; i++){
        std::cout << std::setw(10) << ScanHost[i];
      }
      std::cout << std::endl;
      copyToHost(q,usm_scan_temp,ScanHost);
      std::cout << "EdgeVal Step4 ";
      for(int i=0; i<128; i++){
        std::cout << std::setw(10) << ScanHost[i];
      }
      std::cout << std::endl;
      */
      #endif

  return e1; 
}


template <int krnl_id>
event parallel_levelgen_kernel(queue &q,
                                const uint32_t Vstart,
                                const uint32_t Vsize,
                                uint8_t *usm_visit_mask,
                                uint8_t *usm_visit,
                                const int level_plus_one,
                                uint32_t *usm_pipe,
                                uint32_t *usm_pipe_size,
                                int* usm_dist
                                 ){
   // Define the work-group size and the number of work-groups
    const size_t local_size = LOCAL_WORK_SIZE;  // Number of work-items per work-group
    const size_t global_size = ((Vsize + local_size - 1) / local_size) * local_size;
    // Setup the range
    nd_range<1> range(global_size, local_size);
        auto e = q.submit([&](handler& h) {
        h.parallel_for<class LevelGen<krnl_id>>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
            const int gid = item.get_global_id(0);    // global id

          if (gid < Vsize) {
              uint8_t vmask = usm_visit_mask[gid];
              if(vmask == 1){
                usm_dist[gid + Vstart] = level_plus_one;  
                usm_visit_mask[gid] = 0;
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,sycl::memory_scope::system, sycl::access::address_space::global_space> atomic_op_global(usm_pipe_size[0]);
                usm_pipe[atomic_op_global.fetch_add(1)] = gid + Vstart;
              }
          }

        });
        });
return e;
}

          

//----------------------------------------------------------
//--breadth first search on GPU
//----------------------------------------------------------
// This function instantiates the vector add kernel, which contains
// a loop that adds up the two summand arrays and stores the result
// into sum. This loop will be unrolled by the specified unroll_factor.
// Template function to handle both types
template<typename vectorT>
void GPURun(int vertexCount, 
                  vectorT &IndexHost,
                  vectorT &OffsetHost,
                  std::vector<uint8_t> &h_visit_mask,
                  std::vector<uint8_t> &h_visit,
                  std::vector<std::vector<int>> &h_distance,
                  int sourceNode,const int num_runs,
                  nlohmann::json &newJsonObj,std::vector<uint32_t> &h_visit_offsets) noexcept(false) {

  try {

  auto Devs = sycl::device::get_devices(info::device_type::gpu);


  // if (Devs.size() < 2) {
  //   std::cout << "Cannot test P2P capabilities, at least two devices are "
  //                "required, exiting."
  //             << std::endl;

  // }

  std::vector<sycl::queue> Queues;
  // Insert not all devices only the required ones for model
  std::transform(Devs.begin(), Devs.begin() + NUM_GPU, std::back_inserter(Queues),
                 [](const sycl::device &D) { return sycl::queue{D,{sycl::property::queue::enable_profiling{},
                                                                   sycl::property::queue::in_order{}
                                                                  }}; 
                                           });
  ////////////////////////////////////////////////////////////////////////
  if (Devs.size() > 1){
  if (!Devs[0].ext_oneapi_can_access_peer(
          Devs[1], sycl::ext::oneapi::peer_access::access_supported)) {
    std::cout << "P2P access is not supported by devices, exiting."
              << std::endl;

  }
  }
    std::cout <<"\n----------------------------------------"<< std::endl;

    std::cout << "Running on devices:" << std::endl;
    for(int i =0; i < Queues.size(); i++){

    std::cout << i << ":\t" << Queues[i].get_device().get_info<sycl::info::device::name>()
              << std::endl;
    }
std::cout <<"----------------------------------------"<< std::endl;
  // Enables Devs[x] to access Devs[y] memory and vice versa.
  if (Devs.size() > 1){
  gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID_i) {
    gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID_j) {
        if (gpuID_i != gpuID_j) {
              Devs[gpuID_i].ext_oneapi_enable_peer_access(Devs[gpuID_j]);
          }
    });
  });
  }




    // Compute kernel execution time
    std::vector<sycl::event> levelEvent(NUM_GPU);
    std::vector<sycl::event> exploreEvent(NUM_GPU);
    sycl::event copybackhostEvent;
    std::vector<double> explore_times(NUM_GPU,0);
    std::vector<double> levelgen_times(NUM_GPU,0);


    std::vector<double> run_times(num_runs,0);
    int zero = 0;
    std::vector<uint32_t*> OffsetDevice(NUM_GPU);
    std::vector<uint32_t*> EdgesDevice(NUM_GPU);
    std::vector<uint8_t*> usm_visit_mask(NUM_GPU);
    std::vector<uint8_t*> usm_visit(NUM_GPU);
    std::vector<uint32_t*> ScanTempDevice(NUM_GPU);
    std::vector<uint32_t*> ScanFullDevice(NUM_GPU);
    std::vector<uint32_t*> usm_local_indptr(NUM_GPU);
    std::vector<uint32_t> h_pipe(vertexCount,0);
    std::vector<uint32_t> h_pipe_count(1,1);

    for(int i =0; i < num_runs; i++){
  
      std::fill(h_pipe_count.begin(), h_pipe_count.end(), 1);
      std::fill(h_pipe.begin(), h_pipe.end(), 0);
      h_pipe[0] = sourceNode;

      uint32_t* usm_pipe_global   = malloc_device<uint32_t>(vertexCount, Queues[0]);
      int* usm_dist    = malloc_device<int>(h_distance.back().size(), Queues[0]); 
    gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
      size_t offsetSize;
      size_t indexSize;
      
      if constexpr (std::is_same_v<vectorT, std::vector<uint32_t>>) {
          offsetSize = OffsetHost.size();
          indexSize = IndexHost.size();
      } else if constexpr (std::is_same_v<vectorT, std::vector<std::vector<uint32_t>>>) {
          offsetSize = OffsetHost[gpuID].size();
          indexSize = IndexHost[gpuID].size();
      }

      OffsetDevice[gpuID]        = malloc_device<uint32_t>(offsetSize, Queues[gpuID]);
      EdgesDevice[gpuID]     = malloc_device<uint32_t>(indexSize, Queues[gpuID]); 
      usm_visit_mask[gpuID]    = malloc_device<uint8_t>((h_visit_offsets[gpuID+1] - h_visit_offsets[gpuID]), Queues[gpuID]); 
      usm_visit[gpuID]    = malloc_device<uint8_t>((h_visit_offsets[gpuID+1] - h_visit_offsets[gpuID]), Queues[gpuID]); 
      const uint32_t scanTempSize = (offsetSize-1 + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE;
      usm_local_indptr[gpuID]   = malloc_device<uint32_t>((offsetSize), Queues[gpuID]);
      const uint32_t scanFullSize = offsetSize; //indexSize;
      ScanTempDevice[gpuID]   = malloc_device<uint32_t>(scanTempSize, Queues[gpuID]);
      ScanFullDevice[gpuID]   = malloc_device<uint32_t>(scanFullSize, Queues[gpuID]);
      if(i==0){
        std::cout << "Allocated " << scanTempSize << " elements for ScanTempDevice buffer." << std::endl;
        std::cout << "Allocated " << scanFullSize << " elements for ScanFullDevice buffer." << std::endl;
      }
      /*  Use memset to initialize the ScanFullDevice to 0s
          This will allow us to get rid of the else if cond
          in kernel ExploreNeighbours3
      */
      Queues[gpuID].memset(ScanFullDevice[gpuID], 0, 1 * sizeof(uint32_t)).wait();
      if constexpr (std::is_same_v<vectorT, std::vector<uint32_t>>) {
          copyToDevice(Queues[gpuID],IndexHost,EdgesDevice[gpuID]);
          copyToDevice(Queues[gpuID],OffsetHost,OffsetDevice[gpuID]);
      } else if constexpr (std::is_same_v<vectorT, std::vector<std::vector<uint32_t>>>) {
          copyToDevice(Queues[gpuID],IndexHost[gpuID],EdgesDevice[gpuID]);
          copyToDevice(Queues[gpuID],OffsetHost[gpuID],OffsetDevice[gpuID]);
      }
  
      Queues[0].memcpy(usm_visit_mask[gpuID], h_visit_mask.data() + h_visit_offsets[gpuID],(h_visit_offsets[gpuID+1] - h_visit_offsets[gpuID])* sizeof(uint8_t));
      Queues[0].memcpy(usm_visit[gpuID], h_visit.data() + h_visit_offsets[gpuID], (h_visit_offsets[gpuID+1] - h_visit_offsets[gpuID])* sizeof(uint8_t));

    });
    uint32_t *frontierCountDevice = malloc_device<uint32_t>(1, Queues[0]);







    copyToDevice(Queues[0],h_pipe,usm_pipe_global);
    copyToDevice(Queues[0],h_distance[i],usm_dist);




    Queues[0].memcpy(frontierCountDevice, &zero, sizeof(uint32_t));  
 

    double start_time = 0;
    double end_time = 0;
    for(int iteration=0; iteration < MAX_NUM_LEVELS; iteration++){
      if((h_pipe_count[0]) == 0){
        break;
      }    
       

      gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
              exploreEvent[gpuID] = parallel_explorer_kernel<gpuID>(Queues[gpuID],h_pipe_count[0],OffsetDevice[gpuID],EdgesDevice[gpuID],usm_pipe_global,usm_local_indptr[gpuID], usm_visit_mask[gpuID],usm_visit[gpuID],ScanTempDevice[gpuID],ScanFullDevice[gpuID],h_visit_offsets[gpuID],h_visit_offsets[gpuID+1]- h_visit_offsets[gpuID]);
       
      });
      gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
           Queues[gpuID].wait();
      });
      /*if(i==0){
        std::vector<uint32_t> ScanHost(128);
        copyToHost(Queues[0],ScanTempDevice[0],ScanHost);
        std::cout << "ScanTemp ";
        for(int i=0; i<128; i++){
          std::cout << std::setw(10) << ScanHost[i];
        }
        std::cout << std::endl;
        copyToHost(Queues[0],ScanFullDevice[0],ScanHost);
        std::cout << "ScanFull ";
        for(int i=0; i<128; i++){
          std::cout << std::setw(10) << ScanHost[i];
        }
        std::cout << std::endl;
      }*/
      gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {

            levelEvent[gpuID] =parallel_levelgen_kernel<gpuID>(Queues[gpuID],h_visit_offsets[gpuID],h_visit_offsets[gpuID+1] - h_visit_offsets[gpuID],usm_visit_mask[gpuID],usm_visit[gpuID],iteration+1,usm_pipe_global,frontierCountDevice,usm_dist);
      });
      gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
            Queues[gpuID].wait();
      });



      copyToHost(Queues[0],frontierCountDevice,h_pipe_count);
      Queues[0].wait();
      copybackhostEvent = Queues[0].memcpy(frontierCountDevice, &zero, sizeof(uint32_t));

      // Capture execution times 
      #if VERBOSE == 1
      gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
      explore_times[gpuID]     += GetExecutionTime(exploreEvent[gpuID]);
      levelgen_times[gpuID]     += GetExecutionTime(levelEvent[gpuID]);
      if(i==0)
        std::cout << std::setw(13) << h_pipe_count[0] << " new nodes after iteration " << std::setw(2) << iteration << std::endl;
      });
      #endif

      
      if(iteration == 0){
           start_time = exploreEvent[0].get_profiling_info<info::event_profiling::command_start>();
      }
      
    }
      
      end_time = copybackhostEvent.get_profiling_info<info::event_profiling::command_end>();
      // end_time = max(levelEvent.get_profiling_info<info::event_profiling::command_end>(),levelEventQ.get_profiling_info<info::event_profiling::command_end>());
      double total_time = (end_time - start_time)* 1e-6; // ns to ms
      run_times[i] = total_time;
      copyToHost(Queues[0],usm_dist,h_distance[i]);
      Queues[0].wait();

    gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
          sycl::free(OffsetDevice[gpuID], Queues[gpuID]);
          sycl::free(EdgesDevice[gpuID], Queues[gpuID]);
          sycl::free(usm_visit[gpuID], Queues[gpuID]);
          sycl::free(usm_visit_mask[gpuID], Queues[gpuID]);
          sycl::free(ScanTempDevice[gpuID], Queues[gpuID]);
          sycl::free(ScanFullDevice[gpuID], Queues[gpuID]);
          sycl::free(usm_local_indptr[gpuID], Queues[gpuID]);
    });

    sycl::free(usm_dist, Queues[0]);
    sycl::free(usm_pipe_global, Queues[0]);

    } // for loop num_runs

    #if VERBOSE == 1
    printExecutionTimes(explore_times, levelgen_times);
    #endif




    // Assume removeAnomalies function and threshold are defined elsewhere
    double threshold = 1.5; // This means we consider points beyond 1.5 standard deviations as anomalies
    // https://bookdown.org/kevin_davisross/probsim-book/normal-distributions.html 87%
    // explanation for threshold : https://chatgpt.com/share/6e64d349-bdd6-4662-99c2-2d265dffd43c
    // Remove anomalies
    std::vector<double> filteredData = removeAnomalies(run_times, threshold);
    std::vector<double> filteredDataNOF = removeAnomaliesnofilter(run_times);

    // Output the filtered data in a formatted table
   
    std::cout <<"\n----------------------------------------"<< std::endl;
    double total_exec_time = std::accumulate(run_times.begin(), run_times.end(), 0.0);
    double avg_time = std::accumulate(run_times.begin(), run_times.end(), 0.0) / run_times.size();
    double total_time_filtered = std::accumulate(filteredData.begin(), filteredData.end(), 0.0) / filteredData.size();
    double minimum_time_filtered = *std::min_element(filteredData.begin(), filteredData.end());
    double total_time_90f = std::accumulate(filteredDataNOF.begin(), filteredDataNOF.end(), 0.0) / filteredDataNOF.size();
    double minimum_time_90f = *std::min_element(filteredDataNOF.begin(), filteredDataNOF.end());

    // Print events and execution times
    printRow("Average (filtered) Time:", formatDouble(total_time_filtered)+ " (ms)");
    printRow("Minimum (filtered) Time:", formatDouble(minimum_time_filtered)+ " (ms)");
    
    printRow("Average (90%) Time:", formatDouble(total_time_90f)+ " (ms)");
    printRow("Minimum (90%) Time:", formatDouble(minimum_time_90f)+ " (ms)");
    
    printRow("Average Execution Time:", formatDouble(avg_time) + " (ms)");
    printRow("Total Execution Time:", formatDouble(total_exec_time) + " (ms)");

    newJsonObj["rawExecutionTimeAll"] = run_times;
    newJsonObj["avgExecutionTime"] = avg_time;
    newJsonObj["avgExecutionTimeFiltered"] = total_time_filtered;
    newJsonObj["minExecutionTimeFiltered"] = minimum_time_filtered;
    newJsonObj["avgExecutionTime90f"] = total_time_90f;
    newJsonObj["minExecutionTime90f"] = minimum_time_90f;
    newJsonObj["totalExecutionTime"] = total_exec_time;
    // The queue destructor is invoked when q passes out of scope.
    // q's destructor invokes q's exception handler on any device exceptions.
  }
  catch (sycl::exception const& e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }
}