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
#define NUM_OF_GPUS 2

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
#define THREADS_PER_BLOCK 256

// Aliases for LSU Control Extension types
// Implemented using template arguments such as prefetch & burst_coalesce
// on the new ext::intel::lsu class to specify LSU style and modifiers
// using PrefetchingLSU = ext::intel::lsu<ext::intel::prefetch<true>,ext::intel::statically_coalesce<false>>;
// using PipelinedLSU = ext::intel::lsu<>;
// using BurstCoalescedLSU = ext::intel::lsu<ext::intel::burst_coalesce<true>,ext::intel::statically_coalesce<false>>;
// using CacheLSU = ext::intel::lsu<ext::intel::burst_coalesce<true>, ext::intel::cache<1024*1024>,ext::intel::statically_coalesce<false>>;
// template <typename vertex_t>
// class Limits {
// public:
//     static vertex_t invalid() {
//         return std::numeric_limits<vertex_t>::max();
//     }

//     static bool is_valid(vertex_t v) {
//         return v != invalid();
//     }
// };

// // Define a custom binary operation
// struct plus_op {
//     template <typename T>
//     T operator()(T a, T b) const {
//         return a + b;
//     }
// };
std::vector<sycl::device> get_two_devices() {
  auto devs = sycl::device::get_devices();
  if (devs.size() == 0)
    throw "No devices available";
  if (devs.size() == 1)
    return {devs[0], devs[0]};
  return {devs[0], devs[1]};
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
// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.

// template <int unroll_factor> class LevelGenerator;
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
                                MyUint1 *usm_visit_mask,
                                MyUint1 *usm_visit)
    {

      // Prepare Data 
      // const int V = prefix_sum.back();
      // we can't pass vectors to the sycl kernel so needs to be integers
      // const int prefix_1 = prefix_sum[1]; 
      // no need for this since it is same as V
      // const int prefix_2 = prefix_sum[2]; 


      // Define the work-group size and the number of work-groups
      const size_t local_size = THREADS_PER_BLOCK;  // Number of work-items per work-group
      const size_t global_size = ((V + local_size - 1) / local_size) * local_size;

      // Setup the range
      nd_range<1> range(global_size, local_size);

          auto e = q.submit([&](handler& h) {
            // Local memory for exclusive sum, shared within the work-group
            // sycl::local_accessor<int> local_sum(local_size, h);
            // sycl::local_accessor<int> local_sum2(local_size, h);
            // sycl::local_accessor<Uint32> local_th_deg(local_size, h);
            // sycl::local_accessor<Uint32> vertices(local_size, h);
            sycl::local_accessor<Uint32> sedges(local_size, h);
            sycl::local_accessor<Uint32> degrees(local_size, h);
          h.parallel_for<class ExploreNeighbours<krnl_id>>(range, [=](nd_item<1> item) {
            const int gid = item.get_global_id(0);    // global id
            const int lid  = item.get_local_id(0); // threadIdx.x
            const int blockIdx  = item.get_group(0); // blockIdx.x
            const int gridDim   = item.get_group_range(0); // gridDim.x
            const int blockDim  = item.get_local_range(0); // blockDim.x


            device_ptr<Uint32> DevicePtr_start(usm_nodes_start);  
            device_ptr<Uint32> DevicePtr_end(usm_nodes_start + 1);  
            device_ptr<Uint32> DevicePtr_edges(usm_edges);

            Uint32 v;
            Uint32 local_th_deg; // this variable is shared between workitems
          // this variable will be instantiated for each work-item separately

          // Determine which pipe to read from based on gid
          //  if (gid < V) {
          //         v = usm_pipe_1[gid];
          //         sedges[lid] = DevicePtr_start[v];
          //         local_th_deg =DevicePtr_end[v] - DevicePtr_start[v];
          //     }
          //     else {
          //         local_th_deg = 0;
          //     }

          if (gid < V) {
              // Read from usm_pipe
              v = usm_pipe_1[gid];
              sedges[lid] = DevicePtr_start[v]; // Store in sedges at the correct global index
              local_th_deg = DevicePtr_end[v] - DevicePtr_start[v]; // Assuming this is how you're calculating degree
          }  else {
              local_th_deg = 0;
          }
            // sycl::group_barrier(item.get_group());

            // 2. Exclusive sum of degrees to find total work items per block.
            Uint32 th_deg = sycl::exclusive_scan_over_group(item.get_group(), local_th_deg, sycl::plus<>());
            degrees[lid] = th_deg;
            // sycl::group_barrier(item.get_group());

            // 3. Cumulative sum of total number of nonzeros 
            Uint32 total_nnz = reduce_over_group(item.get_group(), local_th_deg, sycl::plus<>());
            Uint32 length = (V < gid - lid + blockDim) ? (V - (gid -lid)) : blockDim;
      

    for (int i = lid;            // threadIdx.x
        i < total_nnz;  // total degree to process
        i += blockDim    // increment by blockDim.x
    ) {

    /// 4. Compute. Using binary search, find the source vertex each thread is
    /// processing, and the corresponding edge, neighbor and weight tuple. Passed
    /// to the user-defined lambda operator to process. If there's an output, the
    /// resultant neighbor or invalid vertex is written to the output frontier.
      // Implement a simple upper_bound algorithm for use in SYCL
    
      Uint32 it = upper_bound(degrees,length, i);
      Uint32 id =  it - 1;
      Uint32  e = sedges[id] + i  - degrees[id]; 
      Uint32  n  = usm_edges[e];   

      if(!usm_visit[n]){
        usm_visit_mask[n] = 1;
        // usm_visit[n] = 1;
        // usm_dist[n] = iteration + 1;  
      }
    } 
          
      });
          });


  return e; 
}


template <int krnl_id>
event parallel_levelgen_kernel(queue &q,
                                int V,
                                MyUint1 *usm_visit_mask,
                                MyUint1 *usm_visit,
                                int iteration,
                                Uint32 *usm_pipe,
                                Uint32 *usm_pipe_size,
                                Uint32* usm_dist
                                 ){
   // Define the work-group size and the number of work-groups
    const size_t local_size = THREADS_PER_BLOCK;  // Number of work-items per work-group
    const size_t global_size = ((V + local_size - 1) / local_size) * local_size;

    // Setup the range
    nd_range<1> range(global_size, local_size);
        auto e = q.submit([&](handler& h) {
          // Local memory for exclusive sum, shared within the work-group
          // sycl::local_accessor<int> localVisitMask(local_size, h);
          // sycl::local_accessor<int> local_sum(local_size, h);
          // sycl::local_accessor<int> local_sum2(local_size, h);
          // sycl::local_accessor<int> local_counter(local_size, h);
                    // sycl::local_accessor<Uint32> degrees(local_size, h);
                    // sycl::local_accessor<Uint32> vertices(local_size, h);
          // sycl::local_accessor<int> local_counter(1, h);
        h.parallel_for<class LevelGen<krnl_id>>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
          const int gid = item.get_global_id(0);    // global id
          const int lid  = item.get_local_id(0); // threadIdx.x
          const int blockIdx  = item.get_group(0); // blockIdx.x
          const int gridDim   = item.get_group_range(0); // gridDim.x
          const int blockDim  = item.get_local_range(0); // blockDim.x


    // Load a chunk of usm_visit_mask into local memory
        if (gid < V) {
            MyUint1 vmask = usm_visit_mask[gid];
            if(vmask == 1){

              usm_dist[gid] = iteration + 1;  
                usm_visit[gid] = 1;
                                sycl::atomic_ref<Uint32, sycl::memory_order::relaxed,
            sycl::memory_scope::device, 
            sycl::access::address_space::global_space> atomic_op_global(usm_pipe_size[0]);
            
                usm_pipe[atomic_op_global.fetch_add(1)] = gid;
                usm_visit_mask[gid]=0;
            
                // local_counter[lid] = 1;
                // vertices[lid] = gid;
            }
        }
  //           else{
  //             vertices[lid] = -1;
  //             local_counter[lid] = 0;
  //           }
            
  //       }else{
  //         vertices[lid] = -1;
  //         local_counter[lid] = 0;
  //       }
  //       group_barrier(item.get_group());




  //           Uint32 th_deg = local_counter[lid];
  //           //  total_nnz;
 
  //          /// 2. Exclusive sum of degrees to find total work items per block.
  //            th_deg = sycl::exclusive_scan_over_group(item.get_group(), th_deg, sycl::plus<>());

  //           local_sum[lid] = th_deg;

  //           if(lid == blockDim -1)
  //           th_deg +=local_counter[lid];
            
  //           local_sum2[lid] = th_deg;
  //           sycl::group_barrier(item.get_group());

  //           unsigned int total_nnz =  local_sum2[blockDim - 1];
  //           degrees[lid] = local_sum[lid];
  //       /// 3. Compute block offsets if there's an output frontier.
  //         group_barrier(item.get_group());
  //           // Define the atomic operation reference outside the loop
  //         // printf("vertices = %d, lid = %d, degrees[lid] = %d, agg = %d\n", vertices[lid],lid, degrees[lid], total_nnz);

  //         for (int i = lid;            // threadIdx.x
  //      i < total_nnz;  // total degree to process
  //      i += blockDim    // increment by blockDim.x
  // ) {  

  //        auto it = std::upper_bound(degrees.begin(),degrees.end(), i);
  //     // int id = std::distance(degrees, it) - 1;
  //   Uint32 id = std::distance(degrees.begin(), it)-1; // Return the distance minus 1
  //   if(id < degrees.size()){
  //           Uint32 v = vertices[id];
  //               usm_dist[v] = iteration + 1;  
                
                
  //               // Use atomic operation to update the global Frontier array
  //               // sycl::atomic_ref<Uint32, sycl::memory_order::relaxed,
  //               //     sycl::memory_scope::device, 
  //               //     sycl::access::address_space::global_space> atomic_op_global(usm_pipe_size[0]);

  //               // Use atomic operation to update the global Frontier array
  //               sycl::atomic_ref<Uint32, sycl::memory_order::relaxed,
  //           sycl::memory_scope::device, 
  //           sycl::access::address_space::global_space> atomic_op_global(usm_pipe_size[0]);
  //           if(local_counter[id]){
  //               Frontier[atomic_op_global.fetch_add(1)] = v;
  //               usm_visit_mask[v]=0;
  //           }
  //   }
  // }
        // group_barrier(item.get_group());
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
void GPURun(int vertexCount, 
                  std::vector<std::vector<Uint32>> &IndexHost,
                  std::vector<std::vector<Uint32>> &OffsetHost,
                  std::vector<MyUint1> &VisitMaskHost,
                  std::vector<MyUint1> &VisitHost,
                  std::vector<Uint32> &DistanceHost,
                  int sourceNode,int edgeCount,const int num_runs) noexcept(false) {
 

  // Select either:
  //  - the FPGA emulator device (CPU emulation of the FPGA)
  //  - the FPGA device (a real FPGA)
// #if (DEVICE == FPGA_EMULATOR)
  // auto device_selector = sycl::ext::intel::fpga_emulator_selector_v;
// #elif (DEVICE == FPGA_DEVICE)
  // auto device_selector = sycl::ext::intel::fpga_selector_v;
#if (DEVICE == NVIDIA_GPU)
  auto device_selector = sycl::gpu_selector_v;
#endif



  sycl::queue q{device_selector,
                sycl::property::queue::enable_profiling{}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;




  try {
  //   auto devs = sycl::device::get_devices(info::device_type::gpu);

  //   auto Queues[0] = sycl::queue{devs[0],
  //               sycl::property::queue::enable_profiling{}};
  //   auto Queues[1] = sycl::queue{devs[1],
  //               sycl::property::queue::enable_profiling{}}; // if only one device is found, both queues
  //                                   // will use same device

  //  std::cout << "Running on devices:" << std::endl;
  //   std::cout << "1:\t" << Queues[0].get_device().get_info<sycl::info::device::name>()
  //             << std::endl;
  //   std::cout << "2:\t" << Queues[1].get_device().get_info<sycl::info::device::name>()
  //             << std::endl;



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

  // Enables Devs[0] to access Devs[1] memory.
    // Devs[0].ext_oneapi_enable_peer_access(Devs[1]);
    // Devs[1].ext_oneapi_enable_peer_access(Devs[0]);

fpga_tools::UnrolledLoop<NUM_OF_GPUS>([&](auto gpuID_i) {
  fpga_tools::UnrolledLoop<NUM_OF_GPUS>([&](auto gpuID_j) {
      if (gpuID_i != gpuID_j) {
            Devs[gpuID_i].ext_oneapi_enable_peer_access(Devs[gpuID_j]);
        }
  });
});


    // Compute kernel execution time
    std::vector<sycl::event> levelEvent(NUM_OF_GPUS);
    std::vector<sycl::event> exploreEvent(NUM_OF_GPUS);
    sycl::event levelEventQ,exploreEventQ;
    sycl::event pipeEvent,resetEvent;
    sycl::event copybackhostEvent;
    double exploreDuration=0,levelDuration=0;
    double exploreDurationQ=0,levelDurationQ=0;
    double pipeDuration=0,resetDuration=0;


    std::vector<std::vector<Uint32>> distances(num_runs,DistanceHost);
    std::vector<double> run_times(num_runs,0);
    int zero = 0;
    std::vector<Uint32*> OffsetDevice(NUM_OF_GPUS);
    std::vector<Uint32*> EdgesDevice(NUM_OF_GPUS);
    std::vector<MyUint1*> VisitMaskDevice(NUM_OF_GPUS);
    std::vector<MyUint1*> VisitDevice(NUM_OF_GPUS);
    for(int i =0; i < num_runs; i++){
      // Frontier Start
      std::vector<Uint32> frontierCountHostQ1(1,1);
      std::vector<Uint32> frontierCountHostQ2(1,1);

      std::vector<Uint32> FrontierHostQ1(vertexCount,0);
      FrontierHostQ1[0] = sourceNode;
      std::vector<Uint32> FrontierHostQ2(vertexCount,0);
      FrontierHostQ2[0] = sourceNode;



      
      Uint32 *usm_pipe_global = malloc<Uint32>(vertexCount, Queues[0], usm::alloc::device);


    // // Create a vector to store the USM pointers
    // std::vector<Uint32*> usm_pipes_Q1;

    // // Add the USM pointers to the vector
    // usm_pipes_Q1.push_back(usm_pipe_1);
    // usm_pipes_Q1.push_back(usm_pipe_2);

    // for(int gpuID =0; gpuID < num_gpus; gpuID++){
    fpga_tools::UnrolledLoop<NUM_OF_GPUS>([&](auto gpuID) {
    OffsetDevice[gpuID]        = malloc_device<Uint32>(OffsetHost[gpuID].size(), Queues[gpuID]);
    EdgesDevice[gpuID]     = malloc_device<Uint32>(IndexHost[gpuID].size(), Queues[gpuID]); 
    VisitMaskDevice[gpuID]    = malloc_device<MyUint1>(VisitMaskHost.size(), Queues[gpuID]); 
    VisitDevice[gpuID]    = malloc_device<MyUint1>(VisitHost.size(), Queues[gpuID]); 

    copyToDevice(Queues[gpuID],IndexHost[gpuID],EdgesDevice[gpuID]);
    copyToDevice(Queues[gpuID],OffsetHost[gpuID],OffsetDevice[gpuID]);
    copyToDevice(Queues[gpuID],VisitMaskHost,VisitMaskDevice[gpuID]);
    copyToDevice(Queues[gpuID],VisitHost,VisitDevice[gpuID]);

    });
    Uint32 *frontierCountDevice = malloc_device<Uint32>(1, Queues[0]);
    
    Uint32* DistanceDevice    = malloc_device<Uint32>(DistanceHost.size(), Queues[0]); 




    // OffsetDevice[1]        = malloc_device<Uint32>(OffsetHost[1].size(), Queues[1]);
    // Uint32 *EdgesDeviceQ         = malloc_device<Uint32>(IndexHost[1].size(), Queues[1]); 
    // Uint32 *DistanceDeviceQ      = malloc_device<Uint32>(DistanceHost.size(), Queues[1]); 
    // MyUint1 *VisitMaskDeviceQ    = malloc_device<MyUint1>(VisitMaskHost.size(), Queues[1]); 
    // MyUint1 *VisitDeviceQ        = malloc_device<MyUint1>(VisitHost.size(), Queues[1]); 
    
    copyToDevice(Queues[0],FrontierHostQ1,usm_pipe_global);
    copyToDevice(Queues[0],distances[i],DistanceDevice);




    Queues[0].memcpy(frontierCountDevice, &zero, sizeof(Uint32));  
    double start_time = 0;
    double end_time = 0;
    for(int iteration=0; iteration < MAX_NUM_LEVELS; iteration++){
      if((frontierCountHostQ1[0]) == 0){
        // std::cout << "total number of iterations" << iteration << "\n";
        break;
      }    
       

      // for(int gpuID = 0; gpuID < num_gpus; gpuID++){
      // exploreEvent[gpuID] = parallel_explorer_kernel<0>(Queues[gpuID],frontierCountHostQ1[gpuID],iteration,OffsetDevice[gpuID],EdgesDevice[gpuID],usm_pipe_global, VisitMaskDevice[gpuID],VisitDevice[gpuID]);
      // }

      fpga_tools::UnrolledLoop<NUM_OF_GPUS>([&](auto gpuID) {
              exploreEvent[gpuID] = parallel_explorer_kernel<gpuID>(Queues[gpuID],frontierCountHostQ1[0],iteration,OffsetDevice[gpuID],EdgesDevice[gpuID],usm_pipe_global, VisitMaskDevice[gpuID],VisitDevice[gpuID]);
      });
      fpga_tools::UnrolledLoop<NUM_OF_GPUS>([&](auto gpuID) {
            Queues[gpuID].wait();
      });
      fpga_tools::UnrolledLoop<NUM_OF_GPUS>([&](auto gpuID) {
            levelEvent[gpuID] =parallel_levelgen_kernel<gpuID>(Queues[gpuID],vertexCount,VisitMaskDevice[gpuID],VisitDevice[gpuID],iteration,usm_pipe_global,frontierCountDevice,DistanceDevice);
      });
      fpga_tools::UnrolledLoop<NUM_OF_GPUS>([&](auto gpuID) {
            Queues[gpuID].wait();
      });



      copyToHost(Queues[0],frontierCountDevice,frontierCountHostQ1);
      
      Queues[0].wait();
      copybackhostEvent = Queues[0].memcpy(frontierCountDevice, &zero, sizeof(Uint32));
       
      // Capture execution times 
      // exploreDuration   += GetExecutionTime(exploreEvent[0]);
      // exploreDuration   += GetExecutionTime(exploreEvent[1]);
      // levelDuration     += GetExecutionTime(levelEvent[0]);
      // exploreDurationQ  += GetExecutionTime(exploreEventQ);
      // levelDurationQ    += GetExecutionTime(levelEvent[1]);
      // pipeDuration    += GetExecutionTime(pipeEvent);
      // resetDuration   += GetExecutionTime(resetEvent);
      // Increase the level by 1 
      if(iteration == 0)
      start_time = exploreEvent[0].get_profiling_info<info::event_profiling::command_start>();
    }
      end_time = copybackhostEvent.get_profiling_info<info::event_profiling::command_end>();
      // end_time = max(levelEvent.get_profiling_info<info::event_profiling::command_end>(),levelEventQ.get_profiling_info<info::event_profiling::command_end>());
      double total_time = (end_time - start_time)* 1e-6; // ns to ms
      run_times[i] = total_time;
      copyToHost(Queues[0],DistanceDevice,distances[i]);


    fpga_tools::UnrolledLoop<NUM_OF_GPUS>([&](auto gpuID) {
          sycl::free(OffsetDevice[gpuID], Queues[gpuID]);
          sycl::free(EdgesDevice[gpuID], Queues[gpuID]);
          sycl::free(VisitDevice[gpuID], Queues[gpuID]);
          sycl::free(VisitMaskDevice[gpuID], Queues[gpuID]);
    });

    sycl::free(DistanceDevice, Queues[0]);
    sycl::free(usm_pipe_global, Queues[0]);

    } // for loop num_runs
    // // Add corresponding elements of distances and distancesQ and store in distances
    // for (size_t i = 0; i < num_runs; ++i) {
    //     for (size_t j = 0; j < distances[i].size(); ++j) {
    //         if(distances[i][j] == -1 && distancesQ[i][j] != -1){
    //           // update only if it is not updated with Queues[0]
    //           distances[i][j] = distancesQ[i][j];
    //         }
    //     }
    // }
    // copy VisitDevice back to hostArray
    // Queues[0].memcpy(&DistanceHost[0], DistanceDevice, DistanceHost.size() * sizeof(int));
    // copyToHost(Queues[0],DistanceDevice,DistanceHost);

     DistanceHost = distances[num_runs-1];
    // sycl::free(OffsetDevice, Queues[0]);
    // sycl::free(usm_nodes_end, Queues[0]);
    // sycl::free(EdgesDevice, Queues[0]);
    // sycl::free(DistanceDevice, Queues[0]);
    // sycl::free(VisitDevice, Queues[0]);
    // sycl::free(VisitMaskDevice, Queues[0]);
    // sycl::free(usm_mask, Queues[0]);
    // Check if each distances[i] is equal to DistanceHost
    bool all_match_host = true;
    for(int i =0; i < num_runs; i++){
      if (!std::equal(distances[i].begin(), distances[i].end(), DistanceHost.begin())) {
            all_match_host = false;
            std::cout << "distances[" << i << "] does not match DistanceHost.\n";
        }
    }
     if (all_match_host) {
        std::cout << "All distances vectors match DistanceHost.\n";
    }


    // Check if all distances[i] vectors are identical
    bool all_identical = std::all_of(distances.begin(), distances.end(), 
                                      [&](const std::vector<Uint32>& vec) {
                                          return vec == distances[0];
                                      });
    if (all_identical) {
        std::cout << "All distances vectors are identical.\n";
    } else {
        std::cout << "Not all distances vectors are identical.\n";
    }

    const char separator    = ' ';
    const int nameWidth     = 24;
    const int numWidth      = 24;

      printf(
         "|-------------------------+-------------------------|\n"
         "| # Vertices = %d   | # Edges = %d        |\n"
         "|-------------------------+-------------------------|\n"
         "| Kernel                  |    Wall-Clock Time (ns) |\n"
         "|-------------------------+-------------------------|\n",vertexCount,edgeCount);
         for(int i =0; i < num_runs;i++)std::cout << run_times[i] << std::endl;

  double total_time =std::accumulate(run_times.begin(), run_times.end(), 0.0) / run_times.size();

  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " exploreEvent  : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(exploreDuration*1000) + " (ms) " << "| " << std::endl;
  printf("|-------------------------+-------------------------|\n");
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " pipeEvent    : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(pipeDuration*1000) + " (ms) " << "| " << std::endl;
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " levelEvent   : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(levelDuration*1000) + " (ms) "<< "| "  << std::endl;
  printf("|-------------------------+-------------------------|\n");
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " resetEvent  : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(resetDuration*1000) + " (ms) " << "| " << std::endl;
  printf("|-------------------------+-------------------------|\n");
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Total Execution Time  :" << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(total_time) + " (ms) "<< "| "  << std::endl;
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Throughput = "         << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string((edgeCount/(1000000*total_time*1e-3))) + " (MTEPS)" << "| " << std::endl;;
  printf("|-------------------------+-------------------------|\n");


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