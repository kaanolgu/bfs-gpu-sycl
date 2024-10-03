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


// Intel Compatibility Tool (aka c2s)
// #include <dpct/dpct.hpp>
// #include <dpct/rng_utils.hpp>
// #include <dpct/dpl_utils.hpp>
#include <sycl/sycl.hpp>
// This function returns a vector of two (not necessarily distinct) devices,
// allowing computation to be split across said devices.
#define THREADS_PER_BLOCK 1024

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
  double kernel_time = (end_k - start_k) * 1e-9; // ns to s
  return kernel_time;
}


template <typename T>
int upper_bound(const sycl::local_accessor<T> &arr, int n, int value) {
    int start = 0;
    int i;
    while (start < n) {
        i = start + (n - start) / 2;
        if (value < arr[i]) {
            n = i;
        } else {
            start = i + 1;
        }
    } // end while
    return start;
}

//-------------------------------------------------------------------
//-- initialize Kernel for Exploring the neighbours of next to visit 
//-- nodes
//-------------------------------------------------------------------
template <int krnl_id>
class ExploreNeighbours;
template <int krnl_id>
class LevelGen;

template <int krnl_id>
event parallel_explorer_kernel(queue &q,
                                const Uint32 V,
                                Uint32* usm_nodes_start,
                                Uint32 *usm_edges,
                                Uint32* usm_pipe_1,
                                MyUint1 *usm_visit_mask,
                                MyUint1 *usm_visit,
                                const Uint32 Vstart,
                                const Uint32 Vsize)
    {



      // Define the work-group size and the number of work-groups
      const size_t local_size = THREADS_PER_BLOCK;  // Number of work-items per work-group
      const size_t global_size = ((V + local_size - 1) / local_size) * local_size;

      // Setup the range
      nd_range<1> range(global_size, local_size);

      auto e = q.submit([&](handler& h) {
            // Local memory for exclusive sum, shared within the work-group
            sycl::local_accessor<Uint32> edge_pointer(local_size, h);
            sycl::local_accessor<Uint32> degrees(local_size, h);
          h.parallel_for<class ExploreNeighbours<krnl_id>>(range, [=](nd_item<1> item) {
            const int gid = item.get_global_id(0);    // global id
            const int lid  = item.get_local_id(0); // threadIdx.x
            const int blockDim  = item.get_local_range(0); // blockDim.x

            Uint32 v;
            Uint32 local_node_deg; // this variable is shared between workitems
          // this variable will be instantiated for each work-item separately

           // 1. Read from usm_pipe
          if (gid < V) {
              v = usm_pipe_1[gid];
              edge_pointer[lid] = usm_nodes_start[v]; // Store pointer to the edges start
              local_node_deg = usm_nodes_start[v+1] - usm_nodes_start[v]; // Calculate each nodes partition specific degree
          }  else {
              local_node_deg = 0;
          }
          // 2. Cumulative sum of total number of nonzeros 
          Uint32 total_nnz = reduce_over_group(item.get_group(), local_node_deg, sycl::plus<>());
          Uint32 length = (V < gid - lid + blockDim) ? (V - (gid -lid)) : blockDim;
          // 3. Exclusive sum of degrees to find total work items per block.
          degrees[lid] = sycl::exclusive_scan_over_group(item.get_group(), local_node_deg, sycl::plus<>());
          // the exclusive scan over group does not sync the results so group_barrier is needed here
          sycl::group_barrier(item.get_group());
    for (int i = lid;            // threadIdx.x
        i < total_nnz;  // total degree to process
        i += blockDim    // increment by blockDim.x
    ) {

      // 4. Using grid stride loop with binary search to find the corresponding edges 
      Uint32 it = upper_bound(degrees,length, i);
      Uint32 id =  it - 1;
      Uint32  e = edge_pointer[id] + i  - degrees[id]; 
      Uint32  n  = usm_edges[e];   
      if(!usm_visit[n- Vstart])
      usm_visit_mask[n - Vstart] = 1;
      
    } 
          
      });
          });


  return e; 
}


template <int krnl_id>
event parallel_levelgen_kernel(queue &q,
                                Uint32* Vstart,
                                const Uint32 Vsize,
                                MyUint1 *usm_visit_mask,
                                MyUint1 *usm_visit,
                                const int level_plus_one,
                                Uint32 *usm_pipe,
                                Uint32 *usm_pipe_size,
                                int* usm_dist
                                 ){
   // Define the work-group size and the number of work-groups
    const size_t local_size = THREADS_PER_BLOCK;  // Number of work-items per work-group
    const size_t global_size = ((Vsize + local_size - 1) / local_size) * local_size;

    // Setup the range
    nd_range<1> range(global_size, local_size);
        auto e = q.submit([&](handler& h) {
        h.parallel_for<class LevelGen<krnl_id>>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
            const int gid = item.get_global_id(0);    // global id

          if (gid < Vsize) {
              MyUint1 vmask = usm_visit_mask[gid];
              if(vmask == 1){
                usm_dist[gid + Vstart[krnl_id]] = level_plus_one;  
                usm_visit[gid] = 1;
                usm_visit_mask[gid] = 0;
                sycl::atomic_ref<Uint32, sycl::memory_order::relaxed,sycl::memory_scope::system, sycl::access::address_space::global_space> atomic_op_global(usm_pipe_size[0]);
                usm_pipe[atomic_op_global.fetch_add(1)] = gid + Vstart[krnl_id];
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
                  std::vector<MyUint1> &h_visit_mask,
                  std::vector<MyUint1> &h_visit,
                  std::vector<std::vector<int>> &h_distance,
                  int sourceNode,const int num_runs,
                  nlohmann::json &newJsonObj,std::vector<Uint32> &h_visit_offsets) noexcept(false) {

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
                 [](const sycl::device &D) { return sycl::queue{D,sycl::property::queue::enable_profiling{}}; });
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
    std::vector<Uint32*> OffsetDevice(NUM_GPU);
    std::vector<Uint32*> EdgesDevice(NUM_GPU);
    std::vector<MyUint1*> usm_visit_mask(NUM_GPU);
    std::vector<MyUint1*> usm_visit(NUM_GPU);
    std::vector<Uint32> h_pipe(vertexCount,0);
    std::vector<Uint32> h_pipe_count(1,1);

    for(int i =0; i < num_runs; i++){
  
      std::fill(h_pipe_count.begin(), h_pipe_count.end(), 1);
      std::fill(h_pipe.begin(), h_pipe.end(), 0);
      h_pipe[0] = sourceNode;

      Uint32* usm_pipe_global   = malloc_device<Uint32>(vertexCount, Queues[0]);
      int* DistanceDevice    = malloc_device<int>(h_distance.back().size(), Queues[0]); 
      Uint32* usm_device_offsets   = malloc_device<Uint32>(h_visit_offsets.size(), Queues[0]);
    gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
      size_t offsetSize;
      size_t indexSize;
      
      if constexpr (std::is_same_v<vectorT, std::vector<Uint32>>) {
          offsetSize = OffsetHost.size();
          indexSize = IndexHost.size();
      } else if constexpr (std::is_same_v<vectorT, std::vector<std::vector<Uint32>>>) {
          offsetSize = OffsetHost[gpuID].size();
          indexSize = IndexHost[gpuID].size();
      }

      OffsetDevice[gpuID]        = malloc_device<Uint32>(offsetSize, Queues[gpuID]);
      EdgesDevice[gpuID]     = malloc_device<Uint32>(indexSize, Queues[gpuID]); 
      usm_visit_mask[gpuID]    = malloc_device<MyUint1>((h_visit_offsets[gpuID+1] - h_visit_offsets[gpuID]), Queues[gpuID]); 
      usm_visit[gpuID]    = malloc_device<MyUint1>((h_visit_offsets[gpuID+1] - h_visit_offsets[gpuID]), Queues[gpuID]); 


      if constexpr (std::is_same_v<vectorT, std::vector<Uint32>>) {
          copyToDevice(Queues[gpuID],IndexHost,EdgesDevice[gpuID]);
          copyToDevice(Queues[gpuID],OffsetHost,OffsetDevice[gpuID]);
      } else if constexpr (std::is_same_v<vectorT, std::vector<std::vector<Uint32>>>) {
          copyToDevice(Queues[gpuID],IndexHost[gpuID],EdgesDevice[gpuID]);
          copyToDevice(Queues[gpuID],OffsetHost[gpuID],OffsetDevice[gpuID]);
      }
  
      Queues[0].memcpy(usm_visit_mask[gpuID], h_visit_mask.data() + h_visit_offsets[gpuID],(h_visit_offsets[gpuID+1] - h_visit_offsets[gpuID])* sizeof(MyUint1));
      Queues[0].memcpy(usm_visit[gpuID], h_visit.data() + h_visit_offsets[gpuID], (h_visit_offsets[gpuID+1] - h_visit_offsets[gpuID])* sizeof(MyUint1));

    });
    Uint32 *frontierCountDevice = malloc_device<Uint32>(1, Queues[0]);






    copyToDevice(Queues[0],h_visit_offsets,usm_device_offsets);
    copyToDevice(Queues[0],h_pipe,usm_pipe_global);
    copyToDevice(Queues[0],h_distance[i],DistanceDevice);




    Queues[0].memcpy(frontierCountDevice, &zero, sizeof(Uint32));  
 
    
    double start_time = 0;
    double end_time = 0;
    for(int iteration=0; iteration < MAX_NUM_LEVELS; iteration++){
      if((h_pipe_count[0]) == 0){
        break;
      }    
       

      gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
              exploreEvent[gpuID] = parallel_explorer_kernel<gpuID>(Queues[gpuID],h_pipe_count[0],OffsetDevice[gpuID],EdgesDevice[gpuID],usm_pipe_global, usm_visit_mask[gpuID],usm_visit[gpuID],h_visit_offsets[gpuID],h_visit_offsets[gpuID+1]- h_visit_offsets[gpuID]);
       
      });
      gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
           Queues[gpuID].wait();
      });
      gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
            levelEvent[gpuID] =parallel_levelgen_kernel<gpuID>(Queues[gpuID],usm_device_offsets,h_visit_offsets[gpuID+1] - h_visit_offsets[gpuID],usm_visit_mask[gpuID],usm_visit[gpuID],iteration+1,usm_pipe_global,frontierCountDevice,DistanceDevice);
      });
      gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
            Queues[gpuID].wait();
      });



      copyToHost(Queues[0],frontierCountDevice,h_pipe_count);
      Queues[0].wait();
      copybackhostEvent = Queues[0].memcpy(frontierCountDevice, &zero, sizeof(Uint32));

      // Capture execution times 
      #if VERBOSE == 1
      gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
      explore_times[gpuID]     += GetExecutionTime(exploreEvent[gpuID]);
      levelgen_times[gpuID]     += GetExecutionTime(levelEvent[gpuID]);
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
      copyToHost(Queues[0],DistanceDevice,h_distance[i]);
      Queues[0].wait();

    gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
          sycl::free(OffsetDevice[gpuID], Queues[gpuID]);
          sycl::free(EdgesDevice[gpuID], Queues[gpuID]);
          sycl::free(usm_visit[gpuID], Queues[gpuID]);
          sycl::free(usm_visit_mask[gpuID], Queues[gpuID]);
    });

    sycl::free(DistanceDevice, Queues[0]);
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

    double total_time = std::accumulate(run_times.begin(), run_times.end(), 0.0) / run_times.size();
    double total_time_filtered = std::accumulate(filteredData.begin(), filteredData.end(), 0.0) / filteredData.size();
    double minimum_time_filtered = *std::min_element(filteredData.begin(), filteredData.end());
    double total_time_90f = std::accumulate(filteredDataNOF.begin(), filteredDataNOF.end(), 0.0) / filteredDataNOF.size();
    double minimum_time_90f = *std::min_element(filteredDataNOF.begin(), filteredDataNOF.end());
    
    // Print events and execution times
    printRow("Average (filtered) Time:", formatDouble(total_time_filtered)+ " (ms)");
    printRow("Minimum (filtered) Time:", formatDouble(minimum_time_filtered)+ " (ms)");
    
    printRow("Average (90%) Time:", formatDouble(total_time_90f)+ " (ms)");
    printRow("Minimum (90%) Time:", formatDouble(minimum_time_90f)+ " (ms)");
    
    printRow("Average Execution Time:", formatDouble(total_time) + " (ms)");
 

    newJsonObj["rawExecutionTimeAll"] = run_times;
    newJsonObj["avgExecutionTime"] = total_time;
    newJsonObj["avgExecutionTimeFiltered"] = total_time_filtered;
    newJsonObj["minExecutionTimeFiltered"] = minimum_time_filtered;
    newJsonObj["avgExecutionTime90f"] = total_time_90f;
    newJsonObj["minExecutionTime90f"] = minimum_time_90f;
 
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