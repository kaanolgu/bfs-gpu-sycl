using namespace sycl;
 using namespace std::chrono;
#include <oneapi/dpl/utility> // to get std:: libraries working here: 
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
                                Uint32 iteration,
                                Uint32* usm_nodes_start,
                                Uint32 *usm_edges,
                                Uint32* usm_pipe_1,
                                MyUint1 *usm_visit_mask)
    {



      // Define the work-group size and the number of work-groups
      const size_t local_size = THREADS_PER_BLOCK;  // Number of work-items per work-group
      const size_t global_size = ((V + local_size - 1) / local_size) * local_size;

      // Setup the range
      nd_range<1> range(global_size, local_size);

          auto e = q.submit([&](handler& h) {
            // Local memory for exclusive sum, shared within the work-group
            sycl::local_accessor<Uint32> sedges(local_size, h);
            sycl::local_accessor<Uint32> degrees(local_size, h);
          h.parallel_for<class ExploreNeighbours<krnl_id>>(range, [=](nd_item<1> item) {
            const int gid = item.get_global_id(0);    // global id
            const int lid  = item.get_local_id(0); // threadIdx.x
            const int blockDim  = item.get_local_range(0); // blockDim.x

            Uint32 v;
            Uint32 local_th_deg; // this variable is shared between workitems
          // this variable will be instantiated for each work-item separately

           // 1. Read from usm_pipe
          if (gid < V) {
              
              v = usm_pipe_1[gid];
              sedges[lid] = usm_nodes_start[v]; // Store in sedges at the correct global index
              local_th_deg = usm_nodes_start[v+1] - usm_nodes_start[v]; // Assuming this is how you're calculating degree
          }  else {
              local_th_deg = 0;
          }
            // sycl::group_barrier(item.get_group());


            // sycl::group_barrier(item.get_group());

            // 3. Cumulative sum of total number of nonzeros 
            Uint32 total_nnz = reduce_over_group(item.get_group(), local_th_deg, sycl::plus<>());
            Uint32 length = (V < gid - lid + blockDim) ? (V - (gid -lid)) : blockDim;
      if(total_nnz > 0){
                    // 2. Exclusive sum of degrees to find total work items per block.
            degrees[lid] = sycl::exclusive_scan_over_group(item.get_group(), local_th_deg, sycl::plus<>());
      }

    for (int i = lid;            // threadIdx.x
        i < total_nnz;  // total degree to process
        i += blockDim    // increment by blockDim.x
    ) {

    /// 4. Compute. Using binary search, find the source vertex each thread is
    /// processing, and the corresponding edge, neighbor and weight tuple. Passed
    /// to the user-defined lambda operator to process. If there's an output, the
    /// resultant neighbor or invalid vertex is written to the output frontier.
    
      Uint32 it = upper_bound(degrees,length, i);
      Uint32 id =  it - 1;
      Uint32  e = sedges[id] + i  - degrees[id]; 
      Uint32  n  = usm_edges[e];   

      
        usm_visit_mask[n] = 1;
      
    } 
          
      });
          });


  return e; 
}


template <int krnl_id>
event parallel_levelgen_kernel(queue &q,
                                const Uint32 Vstart,
                                const Uint32 Vsize,
                                MyUint1 *usm_visit_mask,
                                MyUint1 *usm_visit,
                                const int iteration,
                                Uint32 *usm_pipe,
                                Uint32 *usm_pipe_size,
                                Uint32* usm_dist
                                 ){
   // Define the work-group size and the number of work-groups
    const size_t local_size = THREADS_PER_BLOCK;  // Number of work-items per work-group
    const size_t global_size = ((Vsize + local_size - 1) / local_size) * local_size;

    // Setup the range
    nd_range<1> range(global_size, local_size);
        auto e = q.submit([&](handler& h) {
            // Local memory for exclusive sum, shared within the work-group
            sycl::local_accessor<Uint32> sedges(local_size, h);
            sycl::local_accessor<Uint32> degrees(local_size, h);

        h.parallel_for<class LevelGen<krnl_id>>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
            const int gid = item.get_global_id(0);    // global id
            const int lid  = item.get_local_id(0); // threadIdx.x
            const int blockDim  = item.get_local_range(0); // blockDim.x
            // const int gidX = item.get_global_id(0) + Vstart;    // global id

            Uint32 local_th_deg; // this variable is shared between workitems
      
          if (gid < Vsize) {
              MyUint1 vmask = usm_visit_mask[gid + Vstart] && !usm_visit[ gid + Vstart];
              if(vmask == 1){
                usm_dist[gid + Vstart] = iteration + 1;  
                usm_visit[gid + Vstart] = 1;
                sycl::atomic_ref<Uint32, sycl::memory_order::relaxed,sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_op_global(usm_pipe_size[0]);
                usm_pipe[atomic_op_global.fetch_add(1)] = gid + Vstart;
              }
          }
    //       if (gid < Vsize) {
    //           sedges[lid] =  gid + Vstart; // Store in sedges at the correct global index
    //           local_th_deg = usm_visit_mask[ gid + Vstart]  && !usm_visit[ gid + Vstart]; // Assuming this is how you're calculating degree
    //       }  else {
    //           local_th_deg = 0;
    //       }



    //         // 3. Cumulative sum of total number of nonzeros 
    //         Uint32 total_nnz = reduce_over_group(item.get_group(), local_th_deg, sycl::plus<>());
    //         Uint32 length = (Vsize < gid - lid + blockDim) ? (Vsize - (gid -lid)) : blockDim;
    


    //         Uint32 temp_pipe_value = 0;
    //         Uint32 old_pipe_size;
    //         if(lid == 0 && total_nnz > 0){
    //           // reserve a section in usm_pipe from other GPUs to write only that section
    //           sycl::atomic_ref<Uint32, sycl::memory_order::relaxed,sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_op_global(usm_pipe_size[0]);
    //           temp_pipe_value = atomic_op_global.fetch_add(total_nnz);  
    //         }
            
    //         // check if in the work-group if we have non-zeros if that work group is all 0's then no need to do extra work
    //         // (initial and very last levels)
    //       if(total_nnz > 0){

    //         // 2. Exclusive sum of degrees to find total work items per block.
    //         degrees[lid] = sycl::exclusive_scan_over_group(item.get_group(), local_th_deg, sycl::plus<>());

    //         // this is same value for all work items so no need to have it in shared local accessor
    //         old_pipe_size = reduce_over_group(item.get_group(), temp_pipe_value, sycl::plus<>());
    //       }

            

    // for (int i = lid;            // threadIdx.x
    //     i < total_nnz;  // total degree to process
    //     i += blockDim    // increment by blockDim.x
    // ) {

    // /// 4. Compute. Using binary search, find the source vertex each thread is
    // /// processing, and the corresponding edge, neighbor and weight tuple. Passed
    // /// to the user-defined lambda operator to process. If there's an output, the
    // /// resultant neighbor or invalid vertex is written to the output frontier.
    
    //   Uint32 it = upper_bound(degrees,length, i);
    //   Uint32 id =  it - 1;
    //   Uint32  e = sedges[id] + i  - degrees[id]; 
  
    //   usm_dist[e] = iteration + 1; 
    //   usm_visit[e] = 1;
    //   usm_pipe[old_pipe_size + i ] = e;

    // } 




          // if (gid < V) {
          //     MyUint1 vmask = usm_visit_mask[gid];
          //     if(vmask == 1){
          //       usm_dist[gid] = iteration + 1;  
          //       usm_visit[gid] = 1;
          //       sycl::atomic_ref<Uint32, sycl::memory_order::relaxed,sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_op_global(usm_pipe_size[0]);
          //       usm_pipe[atomic_op_global.fetch_add(1)] = gid;
          //       usm_visit_mask[gid]=0;
          //     }
          // }
          
          
          // usm_pipe_size[0] = 0;

        });
        });
return e;
}

          

//----------------------------------------------------------
//--breadth first search on FPGA
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
                  std::vector<Uint32> &DistanceHost,
                  int sourceNode,const int num_runs,
                  nlohmann::json &newJsonObj,std::vector<Uint32> &test) noexcept(false) {

  try {

  auto Devs = sycl::device::get_devices(info::device_type::gpu);


  if (Devs.size() < 2) {
    std::cout << "Cannot test P2P capabilities, at least two devices are "
                 "required, exiting."
              << std::endl;

  }

  std::vector<sycl::queue> Queues;
  std::transform(Devs.begin(), Devs.end(), std::back_inserter(Queues),
                 [](const sycl::device &D) { return sycl::queue{D,sycl::property::queue::enable_profiling{}}; });
  ////////////////////////////////////////////////////////////////////////

  if (!Devs[0].ext_oneapi_can_access_peer(
          Devs[1], sycl::ext::oneapi::peer_access::access_supported)) {
    std::cout << "P2P access is not supported by devices, exiting."
              << std::endl;

  }

    std::cout << "Running on devices:" << std::endl;
    for(int i =0; i < Queues.size(); i++){

    std::cout << i << ":\t" << Queues[i].get_device().get_info<sycl::info::device::name>()
              << std::endl;
    }

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

    // std::vector<int> test(4); 
    // test[0] = 0;
    // test[1] = 529448;
    // test[2] = 1090184;
    // test[3] = 2074257;


  // std::vector<std::vector<MyUint1>> h_visit_mask(NUM_GPU);
  // h_visit_mask[0].resize(529448,0);
  // h_visit_mask[1].resize(1090184- 529448,0);
  // h_visit_mask[2].resize(2074257 - 1090184,0);
  // std::vector<std::vector<MyUint1>> h_visit(NUM_GPU);
  // h_visit[0].resize(529448,0);
  // h_visit[1].resize(1090184- 529448,0);
  // h_visit[2].resize(2074257 - 1090184,0);
  // h_visit[0][sourceNode]=1;

    // Compute kernel execution time
    std::vector<sycl::event> levelEvent(NUM_GPU);
    std::vector<sycl::event> exploreEvent(NUM_GPU);
    sycl::event copybackhostEvent;
    // double levelDuration=0;

  // std::vector<MyUint1> h_visit0(colsCount,0); 
  // std::vector<MyUint1> h_visit1(colsCount,0); 

    std::vector<std::vector<Uint32>> distances(num_runs,DistanceHost);
    std::vector<double> run_times(num_runs,0);
    int zero = 0;
    std::vector<Uint32*> OffsetDevice(NUM_GPU);
    std::vector<Uint32*> EdgesDevice(NUM_GPU);
    std::vector<MyUint1*> usm_visit_mask(NUM_GPU);
    std::vector<MyUint1*> usm_visit(NUM_GPU);
    std::vector<Uint32> h_pipe(vertexCount,0);
    std::vector<Uint32> h_pipe_count(1,1);
    for(int i =0; i < num_runs; i++){
      // Frontier Start
      

      std::fill(h_pipe_count.begin(), h_pipe_count.end(), 1);
      std::fill(h_pipe.begin(), h_pipe.end(), 0);
      h_pipe[0] = sourceNode;

      Uint32* usm_pipe_global   = malloc_device<Uint32>(vertexCount, Queues[0]);
      Uint32* DistanceDevice    = malloc_device<Uint32>(DistanceHost.size(), Queues[0]); 

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
      usm_visit_mask[gpuID]    = malloc_device<MyUint1>(h_visit_mask.size(), Queues[gpuID]); 
      usm_visit[gpuID]    = malloc_device<MyUint1>(h_visit.size(), Queues[gpuID]); 


      if constexpr (std::is_same_v<vectorT, std::vector<Uint32>>) {
          copyToDevice(Queues[gpuID],IndexHost,EdgesDevice[gpuID]);
          copyToDevice(Queues[gpuID],OffsetHost,OffsetDevice[gpuID]);
      } else if constexpr (std::is_same_v<vectorT, std::vector<std::vector<Uint32>>>) {
          copyToDevice(Queues[gpuID],IndexHost[gpuID],EdgesDevice[gpuID]);
          copyToDevice(Queues[gpuID],OffsetHost[gpuID],OffsetDevice[gpuID]);
      }
  
      copyToDevice(Queues[gpuID],h_visit_mask,usm_visit_mask[gpuID]);
      copyToDevice(Queues[gpuID],h_visit,usm_visit[gpuID]);

    });
    Uint32 *frontierCountDevice = malloc_device<Uint32>(1, Queues[0]);







    copyToDevice(Queues[0],h_pipe,usm_pipe_global);
    copyToDevice(Queues[0],distances[i],DistanceDevice);




    Queues[0].memcpy(frontierCountDevice, &zero, sizeof(Uint32));  
 
    
    double start_time = 0;
    double end_time = 0;
    for(int iteration=0; iteration < MAX_NUM_LEVELS; iteration++){
      if((h_pipe_count[0]) == 0){
        break;
      }    
       

      gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
              exploreEvent[gpuID] = parallel_explorer_kernel<gpuID>(Queues[gpuID],h_pipe_count[0],iteration,OffsetDevice[gpuID],EdgesDevice[gpuID],usm_pipe_global, usm_visit_mask[gpuID]);
      });
      gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
            Queues[gpuID].wait();
      });
      gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
            levelEvent[gpuID] =parallel_levelgen_kernel<gpuID>(Queues[gpuID],test[gpuID],test[gpuID+1] - test[gpuID],usm_visit_mask[gpuID],usm_visit[gpuID],iteration,usm_pipe_global,frontierCountDevice,DistanceDevice);
      });
      gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
            Queues[gpuID].wait();
      });



      copyToHost(Queues[0],frontierCountDevice,h_pipe_count);
      // copyToHost(Queues[0],usm_visit[0],h_visit0);
      // copyToHost(Queues[1],usm_visit[1],h_visit1);
     
      // gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
            // Queues[gpuID].wait();
      // });
      Queues[0].wait();
      copybackhostEvent = Queues[0].memcpy(frontierCountDevice, &zero, sizeof(Uint32));

      // Capture execution times 
      // levelDuration     += GetExecutionTime(levelEvent[0]);
      // Increase the level by 1 
      
      if(iteration == 0){
           start_time = exploreEvent[0].get_profiling_info<info::event_profiling::command_start>();
      }
      
    }
      
      end_time = copybackhostEvent.get_profiling_info<info::event_profiling::command_end>();
      // end_time = max(levelEvent.get_profiling_info<info::event_profiling::command_end>(),levelEventQ.get_profiling_info<info::event_profiling::command_end>());
      double total_time = (end_time - start_time)* 1e-6; // ns to ms
      run_times[i] = total_time;
      copyToHost(Queues[0],DistanceDevice,distances[i]);


    gpu_tools::UnrolledLoop<NUM_GPU>([&](auto gpuID) {
          sycl::free(OffsetDevice[gpuID], Queues[gpuID]);
          sycl::free(EdgesDevice[gpuID], Queues[gpuID]);
          sycl::free(usm_visit[gpuID], Queues[gpuID]);
          sycl::free(usm_visit_mask[gpuID], Queues[gpuID]);
    });

    sycl::free(DistanceDevice, Queues[0]);
    sycl::free(usm_pipe_global, Queues[0]);

    } // for loop num_runs


    // // Find the first and last non-zero indices
    //  size_t ignore_index = sourceNode; // Change this to the index you want to ignore
    // // Find the first and last non-zero indices, ignoring the specified index
    // auto [first_index, last_index] = find_first_last_nonzero(h_visit0, ignore_index);

    // // Display results
    // if (first_index.has_value() && last_index.has_value()) {
    //     std::cout << "First non-zero index (ignoring index " << ignore_index << "): " << *first_index << "\n";
    //     std::cout << "Last non-zero index: " << *last_index << "\n";
    // } else {
    //     std::cout << "No non-zero values found in the array.\n";
    // }
 
    //  // Find the first and last non-zero indices

    // // Find the first and last non-zero indices, ignoring the specified index
    // auto [first_index1, last_index1] = find_first_last_nonzero(h_visit1, ignore_index);

    // // Display results
    // if (first_index1.has_value() && last_index1.has_value()) {
    //     std::cout << "First non-zero index (ignoring index " << ignore_index << "): " << *first_index1 << "\n";
    //     std::cout << "Last non-zero index: " << *last_index1 << "\n";
    // } else {
    //     std::cout << "No non-zero values found in the array.\n";
    // }

    DistanceHost = distances[num_runs-1];
    // Check if each distances[i] is equal to DistanceHost
    bool all_match_host = true;
    for(int i =0; i < num_runs; i++){
      if (!std::equal(distances[i].begin(), distances[i].end(), DistanceHost.begin())) {
            all_match_host = false;
            std::cout << "distances[" << i << "] does not match others on GPU.\n";
        }
    }
     if (all_match_host) {
        std::cout << "All distances vectors match each other on GPU.\n";
    }



// Assume removeAnomalies function and threshold are defined elsewhere
    double threshold = 1.5; // This means we consider points beyond 1.5 standard deviations as anomalies
    // https://bookdown.org/kevin_davisross/probsim-book/normal-distributions.html 87%
    // explanation for threshold : https://chatgpt.com/share/6e64d349-bdd6-4662-99c2-2d265dffd43c
    // Remove anomalies
    std::vector<double> filteredData = removeAnomalies(run_times, threshold);

    // Output the filtered data in a formatted table
    // printHeader("Filtered", "Wall-Clock Time (ms)");
    // int index = 0;
    // for (const auto& time : filteredData) {
    //     printRow("Run #" + std::to_string(index++) + ":", formatDouble(time));
    // }
    printSeparator();
    printRow("Average (filtered) Time", formatDouble(std::accumulate(filteredData.begin(), filteredData.end(), 0.0) / filteredData.size()));
    printSeparator();
    printRow("Minimum (filtered) Time", formatDouble(*std::min_element(filteredData.begin(), filteredData.end())));
    printSeparator();
    // printHeader("Kernel", "Wall-Clock Time (ns)");
    // for (const auto& run_time : run_times) {
    //     std::cout << run_time << std::endl;
    // }

    double total_time = std::accumulate(run_times.begin(), run_times.end(), 0.0) / run_times.size();
    double total_time_filtered = std::accumulate(filteredData.begin(), filteredData.end(), 0.0) / filteredData.size();
    double minimum_time_filtered = *std::min_element(filteredData.begin(), filteredData.end());

    // Print events and execution times
    printRow("Total Execution Time:", formatDouble(total_time) + " (ms)");
    printSeparator();

    newJsonObj["rawExecutionTimeAll"] = run_times;
    newJsonObj["avgExecutionTime"] = total_time;
    newJsonObj["avgExecutionTimeFiltered"] = total_time_filtered;
    newJsonObj["minExecutionTimeFiltered"] = minimum_time_filtered;
    newJsonObj["valid"] = all_match_host;
 
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