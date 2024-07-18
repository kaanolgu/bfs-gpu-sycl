using namespace sycl;
 using namespace std::chrono;

#include <math.h>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <bitset>
#include "functions.hpp"

#include <sycl/sycl.hpp>
// This function returns a vector of two (not necessarily distinct) devices,
// allowing computation to be split across said devices.
#define MAX_THREADS_PER_BLOCK 256
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
// 
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
event parallel_explorer_kernel(queue &q,
                                int no_of_nodes,
                                unsigned int* usm_nodes_start,
                                unsigned int *usm_edges,
                                int *usm_dist,
                                unsigned int* usm_pipe,
                                MyUint1 *usm_updating_mask,
                                MyUint1 *usm_visited)
    {
    
   // Define the work-group size and the number of work-groups
    const size_t local_size = NUMBER_OF_WORKGROUPS;  // Number of work-items per work-group
    const size_t global_size = ((no_of_nodes + local_size - 1) / local_size) * local_size;


    // Setup the range
    nd_range<1> range(global_size, local_size);

        auto e = q.parallel_for<class ExploreNeighbours>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
        int tid = item.get_global_linear_id();
        int lid = item.get_local_id()[0];

        device_ptr<unsigned int> DevicePtr_start(usm_nodes_start);  
        device_ptr<unsigned int> DevicePtr_end(usm_nodes_start + 1);  
        device_ptr<unsigned int> DevicePtr_edges(usm_edges); 
        device_ptr<MyUint1> DevicePtr_visited(usm_visited);  
        if (tid < no_of_nodes) {

          // Read from the pipe
          unsigned int idx = usm_pipe[tid];    
          // Process the current node in tiles      
          unsigned int nodes_start = DevicePtr_start[idx];
          unsigned int nodes_end = DevicePtr_end[idx];
            // Step 1:  Iterate through pipe(frontier) to retrieve number of
            //          neighbours for each node in the pipe. We defined a new
            //          usm_num_of_neighbours which is 
            //          usm_nodes_start[x+1] - usm_nodes_start[x]
          

          // Process the edges of the current nodes
         for (int j = nodes_start; j < nodes_end; j++) {
            int id = DevicePtr_edges[j];
            MyUint1 visited_condition = DevicePtr_visited[id];
            if (!visited_condition) {
                usm_updating_mask[id]=1;

            }
         }
          }
        
        
    });
        

return e;
}


event parallel_levelgen_kernel(queue &q,
                                int no_of_nodes,
                                int *usm_dist,
                                MyUint1 *usm_updating_mask,
                                MyUint1 *usm_visited,
                                int global_level
                                 ){
   // Define the work-group size and the number of work-groups
    const size_t local_size = NUMBER_OF_WORKGROUPS;  // Number of work-items per work-group
    const size_t global_size = ((no_of_nodes + local_size - 1) / local_size) * local_size;

    // Setup the range
    nd_range<1> range(global_size, local_size);

        auto e = q.parallel_for<class LevelGenerator>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
          int tid = item.get_global_linear_id();
          if (tid < no_of_nodes) {
        // auto e =q.single_task<class LevelGenerator<krnl_id>>( [=]() [[intel::kernel_args_restrict]] {
          
          // #pragma unroll 8
          // [[intel::initiation_interval(1)]]
          // for(int tid =no_of_nodes_start; tid < no_of_nodes_end; tid++){
            unsigned int condition = usm_updating_mask[tid];
            if(condition){
              usm_dist[tid] = global_level;
              usm_visited[tid]=1;
           }
          }

        });

return e;
}

event pipegen_kernel(queue &q,
                                int no_of_nodes,
                                unsigned int *usm_pipe,
                                unsigned int *d_pipe_size,
                                MyUint1 *usm_updating_mask
                                 ){
                                  



          
  
    const size_t local_size = NUMBER_OF_WORKGROUPS;  // Number of work-items per work-group
    const size_t global_size = ((no_of_nodes + local_size - 1) / local_size) * local_size;

    // Setup the range
    nd_range<1> range(global_size, local_size);

        auto e = q.parallel_for<class PipeGenerator>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
          int tid = item.get_global_linear_id();
          sycl::atomic_ref<unsigned int, sycl::memory_order_relaxed,
        sycl::memory_scope_device,sycl::access::address_space::global_space>
        atomic_op_global(d_pipe_size[0]);
          if (tid < no_of_nodes) {
            
              char condition = usm_updating_mask[tid];
              if(condition){
                usm_pipe[atomic_op_global.fetch_add(1)] = tid;
                // atomic_op_global+=1; 
              }
            
            } 
        

        });

return e;
}
                                 

event maskremove_kernel(queue &q,
                                int no_of_nodes,
                                MyUint1 *usm_updating_mask
                                 ){
                                  
   // Define the work-group size and the number of work-groups
    const size_t local_size = NUMBER_OF_WORKGROUPS;  // Number of work-items per work-group
    const size_t global_size = ((no_of_nodes + local_size - 1) / local_size) * local_size;

    // Setup the range
    nd_range<1> range(global_size, local_size);

        auto e = q.parallel_for<class MaskRemove>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
        int tid = item.get_global_linear_id();
        if (tid < no_of_nodes) {
            unsigned int condition = usm_updating_mask[tid];
            if(condition){
              usm_updating_mask[tid]=0;  
           }
          }

        });

return e;
}

// initialize device arr with val, if needed set arr[pos] = pos_val
template <typename T>
void initUSMvec(queue &Q, T *usm_arr,std::vector<T> &arr){
  Q.memcpy(usm_arr, arr.data(), arr.size() * sizeof(T));
}
//----------------------------------------------------------
//--breadth first search on GPU
//----------------------------------------------------------
// This function instantiates the vector add kernel, which contains
// a loop that adds up the two summand arrays and stores the result
// into sum. This loop will be unrolled by the specified unroll_factor.
void run_bfs_fpga(Matrix& Graph) noexcept(false) {
 

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


    // Device Data
    int ncus = q.get_device().get_info<info::device::max_compute_units>();

    std::cout<< "number of maximumx compute units" << ncus << "\n";

    unsigned int  *usm_nodes_start    = malloc_device<unsigned int>(Graph.Offset.size(), q);
    int           *usm_dist           = malloc_device<int>(Graph.nodeCount, q); 
    MyUint1       *usm_updating_mask  = malloc_device<MyUint1>(Graph.nodeCount, q); 
    MyUint1       *usm_visited        = malloc_device<MyUint1>(Graph.nodeCount, q); 
    unsigned int  *usm_edges          = malloc_device<unsigned int>(Graph.Position.size(), q); 
    unsigned int  *usm_pipe           = malloc_device<unsigned int>(Graph.nodeCount, q); 
    unsigned int  *usm_pipe_size      = malloc_device<unsigned int>(1, q); 
    unsigned int  *usm_PrefixSum      = malloc_device<unsigned int>(Graph.nodeCount, q); // prefix sum USM


    initUSMvec(q,usm_edges,Graph.Position);
    initUSMvec(q,usm_nodes_start,Graph.Offset);
    initUSMvec(q,usm_dist,Graph.Distance);
    initUSMvec(q,usm_updating_mask,Graph.VisitMask);
    initUSMvec(q,usm_visited,Graph.Visit);
    initUSMvec(q,usm_pipe,Graph.Frontier);

    unsigned int pipe_size = 1;
    unsigned int *d_pipe_size = malloc_device<unsigned int>(1, q);



    // Compute kernel execution time
    event e_explore_1,e_explore_2,e_explore_3,LevelEvent,ExploreEvent,e_explore_4;
    event PipeEvent,ResetEvent,e_remove_2,e_remove_3;
    double time_kernel=0,time_kernel1=0,time_kernel2=0,time_kernel3=0,time_kernel_levelgen=0,time_kernel_levelgen_1=0,time_kernel_pipegen=0,time_kernel_maskreset=0;
    int global_level = 1;    
    
    for(int ijk=0; ijk < 100; ijk++){
        if(pipe_size == 0){
          std::cout << "total number of iterations" << ijk << "\n";
          break;
        }    
        int zero = 0;
        q.memcpy(d_pipe_size, &zero, sizeof(unsigned int)).wait();
        ExploreEvent = parallel_explorer_kernel(q,pipe_size,usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
        q.wait();
        LevelEvent  = parallel_levelgen_kernel(q,numCols,usm_dist,usm_updating_mask,usm_visited,global_level);
        PipeEvent   = pipegen_kernel(q,numCols,usm_pipe, d_pipe_size,usm_updating_mask);
        q.wait();
        ResetEvent =maskremove_kernel(q,numCols,usm_updating_mask);
        q.wait();

        // get new pipe size
        q.memcpy(&pipe_size, d_pipe_size, sizeof(unsigned int)).wait();
 
      
        time_kernel_levelgen += GetExecutionTime(LevelEvent);
        time_kernel_levelgen_1 += GetExecutionTime(ExploreEvent);
        time_kernel_pipegen += GetExecutionTime(PipeEvent);
        time_kernel_maskreset += GetExecutionTime(ResetEvent);
        
        global_level++;
    }





    // copy usm_visited back to hostArray
    q.memcpy(&Graph.Distance[0], usm_dist, Graph.nodeCount * sizeof(int));

    q.wait();
    // sycl::free(usm_nodes_start, q);
    // sycl::free(usm_nodes_end, q);
    // sycl::free(usm_edges, q);
    sycl::free(usm_dist, q);
    sycl::free(usm_visited, q);
    // sycl::free(usm_updating_mask, q);
    // sycl::free(usm_mask, q);
 

    const char separator    = ' ';
    const int nameWidth     = 24;
    const int numWidth      = 24;

      printf(
         "|-------------------------+-------------------------|\n"
         "| # Vertices = %d   | # Edges = %d        |\n"
         "|-------------------------+-------------------------|\n"
         "| Kernel                  |    Wall-Clock Time (ns) |\n"
         "|-------------------------+-------------------------|\n",Graph.nodeCount,Graph.edgeCount);

  // double fpga_execution_time = (max(max(time_kernel,time_kernel1),max(time_kernel2,time_kernel3)) + max(time_kernel_pipegen,max(time_kernel_levelgen,time_kernel_levelgen_1)) +  time_kernel_maskreset);
//  fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>([&](auto krnlID) {
//   std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Explore_1  : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(execTimesExploreRead[krnlID]) + " (s) " << "| " << std::endl;
//  });
  // std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Explore_2  : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(time_kernel1) + " (s) " << "| " << std::endl;
  // std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Explore_3  : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(time_kernel2) + " (s) " << "| " << std::endl;
  // std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Explore_4  : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(time_kernel3) + " (s) " << "| " << std::endl;
  printf("|-------------------------+-------------------------|\n");
  // std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " PipeGen    : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(time_kernel_pipegen) + " (s) " << "| " << std::endl;
  // std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " LevelGen   : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(time_kernel_levelgen) + " (s) "<< "| "  << std::endl;
  // std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " LevelGen_1 : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(time_kernel_levelgen_1) + " (s) "<< "| "  << std::endl;
  // printf("|-------------------------+-------------------------|\n");
  // std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " MaskReset  : " << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(time_kernel_maskreset) + " (s) " << "| " << std::endl;
  printf("|-------------------------+-------------------------|\n");
  // std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Total Time Elapsed  :" << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string(fpga_execution_time) + " (s) "<< "| "  << std::endl;
  // std::cout << "| " << std::left << std::setw(nameWidth) << std::setfill(separator) << " Throughput = "         << "| " << std::setw(numWidth) << std::setfill(separator)  << std::to_string((numEdges/(1000000*fpga_execution_time))) + " (MTEPS)" << "| " << std::endl;;
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