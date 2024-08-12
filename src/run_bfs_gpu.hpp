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
template<int ITEMS_PER_THREAD>
event parallel_explorer_kernel(queue &q,
                                int V, // number of vertices
                                Uint32 iteration,
                                Uint32* usm_nodes_start,
                                Uint32 *usm_edges,
                                Uint32* usm_pipe,
                                Uint32 *usm_pipe_size,
                                MyUint1 *usm_visit_mask,
                                Uint32 *usm_dist,
                                MyUint1 *usm_visit)
    {

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
        h.parallel_for<class ExploreNeighbours>(range, [=](nd_item<1> item) {
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

            if (gid < V) {
                v = usm_pipe[gid];
                sedges[lid] = DevicePtr_start[v];
                local_th_deg =DevicePtr_end[v] - DevicePtr_start[v];
            }
            else {
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
    }
  } 
        
    });
        });
    
return e;
}



event parallel_levelgen_kernel(queue &q,
                                int V,
                                Uint32 *usm_dist,
                                MyUint1 *usm_visit_mask,
                                MyUint1 *usm_visit,
                                int iteration,
                                Uint32 *usm_pipe,
                                Uint32 *usm_pipe_size
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
        h.parallel_for<class LevelGen>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
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

event pipegen_kernel(queue &q,
                                int V,
                                Uint32 *usm_pipe,
                                Uint32 *usm_pipe_size,
                                MyUint1 *usm_visit_mask
                                 ){
                                  



          
  
    const size_t local_size = THREADS_PER_BLOCK;  // Number of work-items per work-group
    const size_t global_size = ((V + local_size - 1) / local_size) * local_size;

    // Setup the range
    nd_range<1> range(global_size, local_size);

        auto e = q.parallel_for<class PipeGenerator>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
          int gid = item.get_global_id();
          sycl::atomic_ref<Uint32, sycl::memory_order_relaxed,
          sycl::memory_scope_device,sycl::access::address_space::global_space>
          atomic_op_global(usm_pipe_size[0]);
      
          // TODO
          // blockscan here to findd the total elements to insert frontier
          // then another grid-stride loop to write frontier 
          if (gid < V) {
              if(usm_visit_mask[gid]){
                // usm_pipe[atomic_op_global.fetch_add(1)] = gid;
                usm_visit_mask[gid]=0;
              }
            
            } 
        

        });

return e;
}
                                 

event maskremove_kernel(queue &q,
                        int V, // number of vertices
                        MyUint1 *usm_visit_mask
                                 ){
                                  
   // Define the work-group size and the number of work-groups
    const size_t local_size = THREADS_PER_BLOCK;  // Number of work-items per work-group
    const size_t global_size = ((V + local_size - 1) / local_size) * local_size;

    // Setup the range
    nd_range<1> range(global_size, local_size);

        auto e = q.parallel_for<class MaskRemove>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
        int gid = item.get_global_id();
        if (gid < V) {
            if(usm_visit_mask[gid]){
              usm_visit_mask[gid]=0;  
           }
          }

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
    auto devs = sycl::device::get_devices(info::device_type::gpu);

    auto Q1 = sycl::queue{devs[0],
                sycl::property::queue::enable_profiling{}};
    auto Q2 = sycl::queue{devs[1],
                sycl::property::queue::enable_profiling{}}; // if only one device is found, both queues
                                    // will use same device

   std::cout << "Running on devices:" << std::endl;
    std::cout << "1:\t" << Q1.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "2:\t" << Q2.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    // Create a queue bound to the chosen device.
    // If the device is unavailable, a SYCL runtime exception is thrown.
    // queue q(device_selector, fpga_tools::exception_handler, prop_list);
    // std::vector<sycl::device> devs = sycl::device::get_devices();
    // std::cout << "Available" << devs.size() << " devices: " << std::endl;
    // for(int i =0; i < devs.size(); i++){
      // std::cout <<i << ":\t" << sycl::queue{devs[i]}.get_device().get_info<sycl::info::device::name>()
              // << std::endl;
    // }

    // auto Q1 = sycl::queue{devs[1]}; // [opencl:cpu:1] Intel(R) OpenCL, AMD EPYC 7742 64-Core Processor OpenCL 3.0 (Build 0) [2023.THREADS_PER_BLOCK.10.0.17_160000]
    // auto q = sycl::queue{devs[2]}; //  [ext_oneapi_cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.2]

    // std::cout << "Running on devices: " << std::endl;
    // std::cout << "1:\t" << Q1.get_device().get_info<sycl::info::device::name>()
    //           << std::endl;
    // std::cout << "2:\t" << q.get_device().get_info<sycl::info::device::name>()
              // << std::endl;
              
    // Print out the device information.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    // FrontierHost[1] = 87;


    // Uint32* OffsetDevice      = malloc_device<Uint32>(OffsetHost.size(), q);
  
    // Uint32 *EdgesDevice       = malloc_device<Uint32>(IndexHost.size(), q); 


    // copyToDevice(q,IndexHost,EdgesDevice);
    // copyToDevice(q,OffsetHost,OffsetDevice);







    // Compute kernel execution time
    sycl::event levelEvent,exploreEvent;
    sycl::event levelEventQ,exploreEventQ;
    sycl::event pipeEvent,resetEvent;
    double exploreDuration=0,levelDuration=0;
    double exploreDurationQ=0,levelDurationQ=0;
    double pipeDuration=0,resetDuration=0;


    std::vector<std::vector<Uint32>> distances(num_runs,DistanceHost);
    std::vector<std::vector<Uint32>> distancesQ(num_runs,DistanceHost);
    std::vector<double> run_times(num_runs,0);
    int zero = 0;
    for(int i =0; i < num_runs; i++){
      // Frontier Start
      std::vector<Uint32> frontierCountHost(1,1);
      std::vector<Uint32> frontierCountHostQ1(1,1);
      std::vector<Uint32> frontierCountHostQ2(1,1);
      std::vector<Uint32> FrontierHost(vertexCount,0);
      std::vector<Uint32> FrontierHostQ1(vertexCount,0);
      std::vector<Uint32> FrontierHostQ2(vertexCount,0);
      FrontierHost[0] = sourceNode;
      // Frontier End
    Uint32 *FrontierDevice      = malloc_device<Uint32>(FrontierHost.size(), Q1); 
    Uint32 *frontierCountDevice = malloc_device<Uint32>(1, Q1);
    Uint32* OffsetDevice        = malloc_device<Uint32>(OffsetHost[0].size(), Q1);
    Uint32 *EdgesDevice         = malloc_device<Uint32>(IndexHost[0].size(), Q1); 
    Uint32 *DistanceDevice      = malloc_device<Uint32>(DistanceHost.size(), Q1); 
    MyUint1 *VisitMaskDevice    = malloc_device<MyUint1>(VisitMaskHost.size(), Q1); 
    MyUint1 *VisitDevice      = malloc_device<MyUint1>(VisitHost.size(), Q1); 

    Uint32 *FrontierDeviceQ      = malloc_device<Uint32>(FrontierHost.size(), Q2); 
    Uint32 *frontierCountDeviceQ = malloc_device<Uint32>(1, Q2);
    Uint32* OffsetDeviceQ        = malloc_device<Uint32>(OffsetHost[1].size(), Q2);
    Uint32 *EdgesDeviceQ         = malloc_device<Uint32>(IndexHost[1].size(), Q2); 
    Uint32 *DistanceDeviceQ      = malloc_device<Uint32>(DistanceHost.size(), Q2); 
    MyUint1 *VisitMaskDeviceQ    = malloc_device<MyUint1>(VisitMaskHost.size(), Q2); 
    MyUint1 *VisitDeviceQ      = malloc_device<MyUint1>(VisitHost.size(), Q2); 
    copyToDevice(Q1,FrontierHost,FrontierDevice);
    copyToDevice(Q1,IndexHost[0],EdgesDevice);
    copyToDevice(Q1,OffsetHost[0],OffsetDevice);
    copyToDevice(Q1,distances[i],DistanceDevice);
    copyToDevice(Q1,VisitMaskHost,VisitMaskDevice);
    copyToDevice(Q1,VisitHost,VisitDevice);

    copyToDevice(Q2,FrontierHost,FrontierDeviceQ);
    copyToDevice(Q2,IndexHost[1],EdgesDeviceQ);
    copyToDevice(Q2,OffsetHost[1],OffsetDeviceQ);
    copyToDevice(Q2,distancesQ[i],DistanceDeviceQ);
    copyToDevice(Q2,VisitMaskHost,VisitMaskDeviceQ);
    copyToDevice(Q2,VisitHost,VisitDeviceQ);
    double start_time = 0;
    double end_time = 0;
    for(int iteration=0; iteration < MAX_NUM_LEVELS; iteration++){
      if(frontierCountHost[0] == 0){
        std::cout << "total number of iterations" << iteration << "\n";
        break;
      }    
      Q1.memcpy(frontierCountDevice, &zero, sizeof(Uint32)).wait();  
      Q2.memcpy(frontierCountDeviceQ, &zero, sizeof(Uint32)).wait();  
      exploreEvent = parallel_explorer_kernel<1>(Q1,frontierCountHost[0],iteration,OffsetDevice,EdgesDevice,FrontierDevice,frontierCountDevice, VisitMaskDevice,DistanceDevice,VisitDevice);
      exploreEventQ = parallel_explorer_kernel<1>(Q2,frontierCountHost[0],iteration,OffsetDeviceQ,EdgesDeviceQ,FrontierDeviceQ,frontierCountDeviceQ, VisitMaskDeviceQ,DistanceDeviceQ,VisitDeviceQ);
      Q1.wait();
      Q2.wait();
      // Level Generate
      levelEvent =parallel_levelgen_kernel(Q1,vertexCount,DistanceDevice,VisitMaskDevice,VisitDevice,iteration,FrontierDevice,frontierCountDevice);
      levelEventQ =parallel_levelgen_kernel(Q2,vertexCount,DistanceDeviceQ,VisitMaskDeviceQ,VisitDeviceQ,iteration,FrontierDeviceQ,frontierCountDeviceQ);
      Q1.wait();
      Q2.wait();

      copyToHost(Q1,frontierCountDevice,frontierCountHostQ1);
      copyToHost(Q2,frontierCountDeviceQ,frontierCountHostQ2);
      Q1.wait();
      Q2.wait();
      copyToHost(Q1,FrontierDevice,FrontierHostQ1);
      copyToHost(Q2,FrontierDeviceQ,FrontierHostQ2);
      Q1.wait();
      Q2.wait();
      for(int itr=0; itr < frontierCountHostQ1[0]; itr++)
      FrontierHost[itr] = FrontierHostQ1[itr];

      for(int itr=frontierCountHostQ1[0]; itr < frontierCountHostQ1[0] + frontierCountHostQ2[0]; itr++)
        FrontierHost[itr] = FrontierHostQ2[itr - frontierCountHostQ1[0]];

      copyToDevice(Q1,FrontierHost,FrontierDevice);
      copyToDevice(Q2,FrontierHost,FrontierDeviceQ);
      Q1.wait();
      Q2.wait();

      frontierCountHost[0] = frontierCountHostQ1[0] + frontierCountHostQ2[0];
      // frontierHost.reserve(frontierCountHost[0])
      // frontierHostQ1.insert(frontierHostQ1.end(), frontierHostQ2.begin(), frontierHostQ2.end());
      // Capture execution times 
      exploreDuration += GetExecutionTime(exploreEvent);
      levelDuration   += GetExecutionTime(levelEvent);
      exploreDurationQ += GetExecutionTime(exploreEventQ);
      levelDurationQ   += GetExecutionTime(levelEventQ);
      // pipeDuration    += GetExecutionTime(pipeEvent);
      // resetDuration   += GetExecutionTime(resetEvent);
      // Increase the level by 1 
      // if(iteration == 0)
      // start_time = exploreEvent.get_profiling_info<info::event_profiling::command_start>();
      // Q2.wait_and_throw();
    }
      // end_time = levelEvent.get_profiling_info<info::event_profiling::command_end>();
      double total_time = (end_time - start_time)* 1e-6; // ns to ms
      run_times[i] = total_time;
      copyToHost(Q1,DistanceDevice,distances[i]);
      copyToHost(Q2,DistanceDeviceQ,distancesQ[i]);




    sycl::free(OffsetDevice, Q1);
    sycl::free(EdgesDevice, Q1);
    sycl::free(DistanceDevice, Q1);
    sycl::free(VisitDevice, Q1);
    sycl::free(VisitMaskDevice, Q1);

    sycl::free(OffsetDeviceQ, Q2);
    sycl::free(EdgesDeviceQ, Q2);
    sycl::free(DistanceDeviceQ, Q2);
    sycl::free(VisitDeviceQ, Q2);
    sycl::free(VisitMaskDeviceQ, Q2);

    } // for loop num_runs
    // Add corresponding elements of distances and distancesQ and store in distances
    for (size_t i = 0; i < num_runs; ++i) {
        for (size_t j = 0; j < distances[i].size(); ++j) {
            if(distances[i][j] == -1 && distancesQ[i][j] != -1){
              // update only if it is not updated with Q1
              distances[i][j] = distancesQ[i][j];
            }
        }
    }
    // copy VisitDevice back to hostArray
    // Q1.memcpy(&DistanceHost[0], DistanceDevice, DistanceHost.size() * sizeof(int));
    // copyToHost(Q1,DistanceDevice,DistanceHost);

     DistanceHost = distances[num_runs-1];
    // sycl::free(OffsetDevice, Q1);
    // sycl::free(usm_nodes_end, Q1);
    // sycl::free(EdgesDevice, Q1);
    // sycl::free(DistanceDevice, Q1);
    // sycl::free(VisitDevice, Q1);
    // sycl::free(VisitMaskDevice, Q1);
    // sycl::free(usm_mask, Q1);
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