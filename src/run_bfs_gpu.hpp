using namespace sycl;
 using namespace std::chrono;

#include <math.h>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <bitset>
#include "functions.hpp"

// Intel Compatibility Tool (aka c2s)
// #include <dpct/dpct.hpp>
// #include <dpct/rng_utils.hpp>
// #include <dpct/dpl_utils.hpp>
#include <sycl/sycl.hpp>
// This function returns a vector of two (not necessarily distinct) devices,
// allowing computation to be split across said devices.
#define NUMBER_OF_WORKGROUPS 64

// Aliases for LSU Control Extension types
// Implemented using template arguments such as prefetch & burst_coalesce
// on the new ext::intel::lsu class to specify LSU style and modifiers
// using PrefetchingLSU = ext::intel::lsu<ext::intel::prefetch<true>,ext::intel::statically_coalesce<false>>;
// using PipelinedLSU = ext::intel::lsu<>;
// using BurstCoalescedLSU = ext::intel::lsu<ext::intel::burst_coalesce<true>,ext::intel::statically_coalesce<false>>;
// using CacheLSU = ext::intel::lsu<ext::intel::burst_coalesce<true>, ext::intel::cache<1024*1024>,ext::intel::statically_coalesce<false>>;


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





//-------------------------------------------------------------------
//-- initialize Kernel for Exploring the neighbours of next to visit 
//-- nodes
//-------------------------------------------------------------------
template<int ITEMS_PER_THREAD>
event parallel_explorer_kernel(queue &q,
                                int V, // number of vertices
                                Uint32* Offset,
                                Uint32 *Edges,
                                Uint32* Frontier,
                                MyUint1 *VisitMask,
                                MyUint1 *Visit,
                                Uint32 *PrefixSum)
    {
    
   // Define the work-group size and the number of work-groups
    const size_t local = NUMBER_OF_WORKGROUPS;  // Number of work-items per work-group
    const size_t global = ((V + local - 1) / local) * local;
    // int ncus = q.get_device().get_info<info::device::max_compute_units>();

    // std::cout<< "number of maximumx compute units" << ncus << "\n";
    // Allocate USM shared memory
    Uint32 *degrees = malloc_device<Uint32>(local, q);
    Uint32 *sedges = malloc_device<Uint32>(local, q);
    Uint32 *vertices = malloc_device<Uint32>(local, q);
    Uint32 *block_offsets = malloc_device<Uint32>(1, q);
    Uint32 *offset = malloc_device<Uint32>(1, q);



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






    // Setup the range
    nd_range<1> range(global, local);

        auto e = q.parallel_for<class ExploreNeighbours>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
          const int globalIdx     = item.get_global_id(0);    // global id
          const int localIdx     = item.get_local_id(0); // threadIdx.x
          const int blockIdx    = item.get_group(0); 
          const int gridDim   = item.get_group_range(0); // gridDim.x
          const int blockDim = item.get_local_range(0); // blockDim.x
          Uint32 th_deg[ITEMS_PER_THREAD]; // this variable is shared between workitems
        // this variable will be instantiated for each work-item separately



        device_ptr<Uint32> DevicePtr_start(Offset);  
        device_ptr<Uint32> DevicePtr_end(Offset + 1);  
        device_ptr<Uint32> DevicePtr_edges(Edges); 
        device_ptr<MyUint1> DevicePtr_visited(Visit);  
        if (globalIdx < V) {
   
          // Process the current node in tiles
          unsigned int v = Frontier[globalIdx];
          vertices[localIdx] = v;      
          sedges[localIdx] = DevicePtr_start[v];
          th_deg[0] = DevicePtr_end[v] - DevicePtr_start[v];
          // degrees[localIdx] = th_deg;
          item.barrier(sycl::access::fence_space::local_space);
        // DEBUG
        // ––––––––
        //  for(int j = 0; j < degrees[localIdx]; j++) {
        //     int iterator =  sedges[localIdx] + j;
        //     int id = DevicePtr_edges[iterator];
        //     MyUint1 visited_condition = DevicePtr_visited[id];
        //     if (!visited_condition) {
        //         VisitMask[id]=1;
        //     }
        //  }

            // Step 1:  Iterate through pipe(frontier) to retrieve number of
            //          neighbours for each node in the pipe. We defined a new
            //          usm_num_of_neighbours which is 
            //          Offset[x+1] - Offset[x]
            //  https://github.com/oneapi-src/oneAPI-samples/blob/bcc0cfe2bf4479303391dc33104f88f57f7d5f73/Libraries/oneMKL/guided_american_options_SYCLmigration/src/longstaff_schwartz_svd_2.cu
          //  CUDA code: 
          //  BlockScan(smem_storage.for_scan).ExclusiveSum(in_the_money, partial_sum, total_sum);
          // 
          //  SYCL equivalent: 
          //  partial_sum = dpct::group::exclusive_scan(item_ct1, in_the_money, 0, sycl::plus<>(), total_sum);
          // 
          
          // int aggregate_degree_per_block;
          // int local_th_deg = th_deg[0];
          // th_deg = dpct::group::exclusive_scan(item, th_deg, 0, sycl::plus<>(), aggregate_degree_per_block);
          // item.barrier(sycl::access::fence_space::local_space);

           // Store back to shared memory (to later use in the binary search).
          // degrees[localIdx] = th_deg[0];
          item.barrier(sycl::access::fence_space::local_space);
                for (int j = 0; j < th_deg[0]; j++) {
                  int iterator =  sedges[localIdx] + j - degrees[localIdx];
            int id = DevicePtr_edges[iterator];
            MyUint1 visited_condition = DevicePtr_visited[id];
            if (!visited_condition) {
                VisitMask[id]=1;

            }
         }

        }



//           if(localIdx == 0){
//             sycl::atomic_ref<Uint32, sycl::memory_order::relaxed, 
//                                      sycl::memory_scope::device, 
//                                      sycl::access::address_space::global_space> atomic_ref(block_offsets[0]);
//                     offset[0] = atomic_ref.fetch_add(aggregate_degree_per_block);
//           }
//           item.barrier(sycl::access::fence_space::local_space);


//           auto length = globalIdx - localIdx + blockDim;

//           length -= globalIdx - localIdx;

//   for (unsigned int i = localIdx;            // threadIdx.x
//        i < aggregate_degree_per_block;  // total degree to process
//        i += blockDim     // increment by blockDim.x
//   ) {
//   /// 4. Compute. Using binary search, find the source vertex each thread is
//   /// processing, and the corresponding edge, neighbor and weight tuple. Passed
//   /// to the user-defined lambda operator to process. If there's an output, the
//   /// resultant neighbor or invalid vertex is written to the output frontier.
//     // Implement a simple upper_bound algorithm for use in SYCL
//     // for(int k =0; k < local; k++){
//     // int left = 0;
//     // int right = length;

//     // while (left < right) {
//     //     int mid = left + (right - left) / 2;
//     //     if (degrees[mid] <= i) {
//     //         left = mid + 1;
//     //     } else {
//     //         right = mid;
//     //     }
//     // }
//     // }
//      auto it = std::upper_bound(degrees, degrees + length, i);
//       int id = std::distance(degrees, it) - 1;
//     // unsigned int id = left - 1; // Return the distance minus 1
    
//     // Read from the frontier
//     Uint32 vv = vertices[id];              // source
//     auto e = sedges[id] + i - degrees[id]; // edge
//     auto n = DevicePtr_edges[e];           // neighbour
//       // if (!Visit[n]) {
//       //  VisitMask[n]=1;
//       // }
// }



//   }




          // Process the edges of the current nodes
        //  for (int j = nodesStart; j < nodesEnd; j++) {
            // int id = DevicePtr_edges[j];
            // MyUint1 visited_condition = DevicePtr_visited[id];
            // if (!visited_condition) {
                // VisitMask[id]=1;

            // }
        //  }
          //  }
        
        
    });
        // });

return e;
}



event parallel_levelgen_kernel(queue &q,
                                int V,
                                int *Distance,
                                MyUint1 *VisitMask,
                                MyUint1 *Visit,
                                int global_level
                                 ){
   // Define the work-group size and the number of work-groups
    const size_t local = NUMBER_OF_WORKGROUPS;  // Number of work-items per work-group
    const size_t global = ((V + local - 1) / local) * local;

    // Setup the range
    nd_range<1> range(global, local);

        auto e = q.parallel_for<class LevelGenerator>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
          int gid = item.get_global_id();
          if (gid < V) {
            if(VisitMask[gid]){
              Distance[gid] = global_level;
              Visit[gid]=1;
           }
          }

        });

return e;
}

event pipegen_kernel(queue &q,
                                int V,
                                Uint32 *Frontier,
                                Uint32 *frontierCount,
                                MyUint1 *VisitMask
                                 ){
                                  



          
  
    const size_t local = NUMBER_OF_WORKGROUPS;  // Number of work-items per work-group
    const size_t global = ((V + local - 1) / local) * local;

    // Setup the range
    nd_range<1> range(global, local);

        auto e = q.parallel_for<class PipeGenerator>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
          int gid = item.get_global_id();
          
          sycl::atomic_ref<Uint32, sycl::memory_order_relaxed,
          sycl::memory_scope_device,sycl::access::address_space::global_space>
          atomic_op_global(frontierCount[0]);
          
          if (gid < V) {
              if(VisitMask[gid]){
                Frontier[atomic_op_global.fetch_add(1)] = gid;
              }
            
            } 
        

        });

return e;
}
                                 

event maskremove_kernel(queue &q,
                        int V, // number of vertices
                        MyUint1 *VisitMask
                                 ){
                                  
   // Define the work-group size and the number of work-groups
    const size_t local = NUMBER_OF_WORKGROUPS;  // Number of work-items per work-group
    const size_t global = ((V + local - 1) / local) * local;

    // Setup the range
    nd_range<1> range(global, local);

        auto e = q.parallel_for<class MaskRemove>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
        int gid = item.get_global_id();
        if (gid < V) {
            if(VisitMask[gid]){
              VisitMask[gid]=0;  
           }
          }

        });

return e;
}

// initialize device arr with val, if needed set arr[pos] = pos_val
template <typename T>
void copyToDevice(queue &Q, std::vector<T> &arr,T *usm_arr){
  Q.memcpy(usm_arr, arr.data(), arr.size() * sizeof(T));
}
// initialize device arr with val, if needed set arr[pos] = pos_val
template <typename T>
void copyToHost(queue &Q, T *usm_arr,std::vector<T> &arr){
  Q.memcpy(arr.data(),usm_arr, arr.size() * sizeof(T)).wait();
}
//----------------------------------------------------------
//--breadth first search on FPGA
//----------------------------------------------------------
// This function instantiates the vector add kernel, which contains
// a loop that adds up the two summand arrays and stores the result
// into sum. This loop will be unrolled by the specified unroll_factor.
void FPGARun(int vertexCount, 
                  std::vector<Uint32> &IndexHost,
                  std::vector<Uint32> &OffsetHost,
                  std::vector<MyUint1> &VisitMaskHost,
                  std::vector<MyUint1> &VisitHost,
                  std::vector<int> &DistanceHost,
                  int sourceNode,int edgeCount) noexcept(false) {
 

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

    // Create a queue bound to the chosen device.
    // If the device is unavailable, a SYCL runtime exception is thrown.
    // queue q(device_selector, fpga_tools::exception_handler, prop_list);
    // std::vector<sycl::device> devs = sycl::device::get_devices();
    // std::cout << "Available" << devs.size() << " devices: " << std::endl;
    // for(int i =0; i < devs.size(); i++){
      // std::cout <<i << ":\t" << sycl::queue{devs[i]}.get_device().get_info<sycl::info::device::name>()
              // << std::endl;
    // }

    // auto Q1 = sycl::queue{devs[1]}; // [opencl:cpu:1] Intel(R) OpenCL, AMD EPYC 7742 64-Core Processor OpenCL 3.0 (Build 0) [2023.NUMBER_OF_WORKGROUPS.10.0.17_160000]
    // auto q = sycl::queue{devs[2]}; //  [ext_oneapi_cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.2]

    // std::cout << "Running on devices: " << std::endl;
    // std::cout << "1:\t" << Q1.get_device().get_info<sycl::info::device::name>()
    //           << std::endl;
    // std::cout << "2:\t" << q.get_device().get_info<sycl::info::device::name>()
              // << std::endl;
              
    // Print out the device information.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::vector<Uint32> FrontierHost(vertexCount,0);
    std::vector<Uint32> PrefixSumHost(vertexCount+1,0);
    FrontierHost[0] = sourceNode;
    Uint32 pipe_size=1;

    Uint32* OffsetDevice      = malloc_device<Uint32>(OffsetHost.size(), q);
    int *DistanceDevice       = malloc_device<int>(DistanceHost.size(), q); 
    MyUint1 *VisitMaskDevice  = malloc_device<MyUint1>(VisitMaskHost.size(), q); 
    MyUint1 *VisitDevice      = malloc_device<MyUint1>(VisitHost.size(), q); 
    Uint32 *EdgesDevice       = malloc_device<Uint32>(IndexHost.size(), q); 
    Uint32 *FrontierDevice    = malloc_device<Uint32>(FrontierHost.size(), q); 
    Uint32 *usm_pipe_size     = malloc_device<Uint32>(1, q); 
    Uint32 *PrefixSumDevice   = malloc_device<Uint32>(PrefixSumHost.size(), q); // prefix sum USM


    copyToDevice(q,IndexHost,EdgesDevice);
    copyToDevice(q,OffsetHost,OffsetDevice);
    copyToDevice(q,DistanceHost,DistanceDevice);
    copyToDevice(q,VisitMaskHost,VisitMaskDevice);
    copyToDevice(q,VisitHost,VisitDevice);
    copyToDevice(q,FrontierHost,FrontierDevice);

    // We will have a single element int the pipe which is source vertex
    std::vector<Uint32> frontierCountHost(1,1);
    Uint32 *frontierCountDevice = malloc_device<Uint32>(1, q);



    // Compute kernel execution time
    sycl::event levelEvent,exploreEvent;
    sycl::event pipeEvent,resetEvent;
    double exploreDuration=0,levelDuration=0;
    double pipeDuration=0,resetDuration=0;



    int global_level = 1;
    int zero = 0;

    for(int ijk=0; ijk < 100; ijk++){
      if(frontierCountHost[0] == 0){
        std::cout << "total number of iterations" << ijk << "\n";
        break;
      }    
      q.memcpy(frontierCountDevice, &zero, sizeof(Uint32)).wait();  
      exploreEvent = parallel_explorer_kernel<1>(q,frontierCountHost[0],OffsetDevice,EdgesDevice,FrontierDevice, VisitMaskDevice,VisitDevice,PrefixSumDevice);
      q.wait();
      // Level Generate
      levelEvent =parallel_levelgen_kernel(q,vertexCount,DistanceDevice,VisitMaskDevice,VisitDevice,global_level);
      pipeEvent =pipegen_kernel(q,vertexCount,FrontierDevice, frontierCountDevice,VisitMaskDevice);
      q.wait();
      resetEvent =maskremove_kernel(q,vertexCount,VisitMaskDevice);             
      q.wait();
      copyToHost(q,frontierCountDevice,frontierCountHost);
      // Capture execution times 
      exploreDuration += GetExecutionTime(exploreEvent);
      levelDuration   += GetExecutionTime(levelEvent);
      pipeDuration    += GetExecutionTime(pipeEvent);
      resetDuration   += GetExecutionTime(resetEvent);
      // Increase the level by 1 
      global_level++;
    }


    // copy VisitDevice back to hostArray
    // q.memcpy(&DistanceHost[0], DistanceDevice, DistanceHost.size() * sizeof(int));
    copyToHost(q,DistanceDevice,DistanceHost);
    q.wait();
    // sycl::free(OffsetDevice, q);
    // sycl::free(usm_nodes_end, q);
    // sycl::free(EdgesDevice, q);
    sycl::free(DistanceDevice, q);
    sycl::free(VisitDevice, q);
    // sycl::free(VisitMaskDevice, q);
    // sycl::free(usm_mask, q);
 

    const char separator    = ' ';
    const int nameWidth     = 24;
    const int numWidth      = 24;

      printf(
         "|-------------------------+-------------------------|\n"
         "| # Vertices = %d   | # Edges = %d     | #Instances = %d   |\n"
         "|-------------------------+-------------------------|\n"
         "| Kernel                  |    Wall-Clock Time (ns) |\n"
         "|-------------------------+-------------------------|\n",vertexCount,edgeCount,(global_level-1));

  double totalDuration = exploreDuration + max(levelDuration,pipeDuration) +  resetDuration; // in seconds

  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " exploreEvent  : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(exploreDuration*1000) + " (ms) " << "| " << std::endl;
  printf("|-------------------------+-------------------------|\n");
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " pipeEvent    : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(pipeDuration*1000) + " (ms) " << "| " << std::endl;
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " levelEvent   : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(levelDuration*1000) + " (ms) "<< "| "  << std::endl;
  printf("|-------------------------+-------------------------|\n");
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " resetEvent  : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(resetDuration*1000) + " (ms) " << "| " << std::endl;
  printf("|-------------------------+-------------------------|\n");
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Total Execution Time  :" << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(totalDuration*1000) + " (ms) "<< "| "  << std::endl;
  std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Throughput = "         << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string((edgeCount/(1000000*totalDuration))) + " (MTEPS)" << "| " << std::endl;;
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