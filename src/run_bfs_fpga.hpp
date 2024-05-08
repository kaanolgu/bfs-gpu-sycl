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

// Aliases for LSU Control Extension types
// Implemented using template arguments such as prefetch & burst_coalesce
// on the new ext::intel::lsu class to specify LSU style and modifiers
// using PrefetchingLSU = ext::intel::lsu<ext::intel::prefetch<true>,ext::intel::statically_coalesce<false>>;
// using PipelinedLSU = ext::intel::lsu<>;
// using BurstCoalescedLSU = ext::intel::lsu<ext::intel::burst_coalesce<true>,ext::intel::statically_coalesce<false>>;
// using CacheLSU = ext::intel::lsu<ext::intel::burst_coalesce<true>, ext::intel::cache<1024*1024>,ext::intel::statically_coalesce<false>>;


// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
template <int unroll_factor> class ExploreNeighbours;
template <int unroll_factor> class LevelGenerator;
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
template <int krnl_id>
event parallel_explorer_kernel(queue &q,
                                int no_of_nodes,
                                unsigned int offset,
                                unsigned int offset_inds,
                                unsigned int* usm_nodes_start,
                                unsigned int *usm_edges,
                                int *usm_dist,
                                unsigned int* usm_pipe,
                                MyUint1 *usm_updating_mask,
                                MyUint1 *usm_visited)
    {
    
   // Define the work-group size and the number of work-groups
    const size_t local_size = 4;  // Number of work-items per work-group
    const size_t global_size = ((no_of_nodes + local_size - 1) / local_size) * local_size;

    // Setup the range
    nd_range<1> range(global_size, local_size);

        auto e = q.parallel_for<class ExploreNeighbours<krnl_id>>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
        int tid = item.get_global_linear_id();
        device_ptr<unsigned int> DevicePtr_start(usm_nodes_start+offset);  
        device_ptr<unsigned int> DevicePtr_end(usm_nodes_start + 1+offset);  
        device_ptr<unsigned int> DevicePtr_edges(usm_edges+offset_inds);  
        device_ptr<MyUint1> DevicePtr_visited(usm_visited);  
        if (tid < no_of_nodes) {

          // Read from the pipe
          unsigned int idx = usm_pipe[tid];    
          // Process the current node in tiles      
          unsigned int nodes_start = DevicePtr_start[idx];
          unsigned int nodes_end = DevicePtr_end[idx];


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


template <int krnl_id>
event parallel_levelgen_kernel(queue &q,
                                int no_of_nodes_start,
                                int no_of_nodes_end,
                                int *usm_dist,
                                MyUint1 *usm_updating_mask,
                                MyUint1 *usm_visited,
                                int global_level
                                 ){
                                  


        auto e =q.single_task<class LevelGenerator<krnl_id>>( [=]() [[intel::kernel_args_restrict]] {
          
          #pragma unroll 8
          [[intel::initiation_interval(1)]]
          for(int tid =no_of_nodes_start; tid < no_of_nodes_end; tid++){
            unsigned int condition = usm_updating_mask[tid];
            if(condition){
              usm_dist[tid] = global_level;
              usm_visited[tid]=1;
           }
          }

        });

return e;
}
template <int unroll_factor>
event pipegen_kernel(queue &q,
                                int no_of_nodes,
                                unsigned int *usm_pipe,
                                unsigned int *d_over,
                                MyUint1 *usm_updating_mask
                                 ){
                                  


        auto e =q.single_task<class PipeGenerator>(  [=]() [[intel::kernel_args_restrict]] {
          int iter = 0;
          int temp[BUFFER_SIZE * 2]; 
          d_type3 temp_pos = 0;
          [[intel::initiation_interval(1)]]
          for(int tid =0; tid < no_of_nodes; tid+=BUFFER_SIZE){
            char condition[BUFFER_SIZE];
            d_type3 increment = 0;
            #pragma unroll
            for(int j = 0; j < BUFFER_SIZE; j++){
              condition[j] = usm_updating_mask[tid + j];
              if(condition[j] && ((tid+j) < no_of_nodes) ){
                increment++;
              }
            } 
            d_type3 current =0;
            #pragma unroll
            for(int j = 0; j < BUFFER_SIZE; j++){
              if(condition[j] && ((tid+j) < no_of_nodes) ){
                temp[temp_pos+current] = tid + j;
                current++;
               
              }
            } 
            temp_pos += increment;
            if(temp_pos >= BUFFER_SIZE){
              #pragma unroll
              for(int j = 0; j < BUFFER_SIZE; j++){
                usm_pipe[iter+j] = temp[j];
              }
              iter += BUFFER_SIZE;
              #pragma unroll
              for(int j = 0; j < BUFFER_SIZE; j++){
                temp[j] = temp[j + BUFFER_SIZE];
              }
              temp_pos-=BUFFER_SIZE;
            }
            // check if the buffer is filled 
            // write buffer back to usm_pipe
          }
          // dump remaining inside the buffer to the output usm_pipe.
          for(int rest = 0; rest < temp_pos; rest++){
            usm_pipe[iter+rest] = temp[rest];
          }

          d_over[0] = iter + temp_pos;

        });

return e;
}
template <int unroll_factor>
event maskremove_kernel(queue &q,
                                int no_of_nodes,
                                MyUint1 *usm_updating_mask
                                 ){
                                  
   // Define the work-group size and the number of work-groups
    const size_t local_size = 16;  // Number of work-items per work-group
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
//--breadth first search on FPGA
//----------------------------------------------------------
// This function instantiates the vector add kernel, which contains
// a loop that adds up the two summand arrays and stores the result
// into sum. This loop will be unrolled by the specified unroll_factor.
template <int unroll_factor>
void run_bfs_fpga(int numCols, 
                  GraphData &host_cu_data,
                  std::vector<unsigned int> &source_inds,
                  std::vector<unsigned int> &source_indptr,
                  std::vector<MyUint1> &h_updating_graph_mask,
                  std::vector<MyUint1> &h_graph_visited,
                  std::vector<int> &h_dist,
                  std::vector<unsigned int> &offset,
                   std::vector<unsigned int> &offset_inds,
                  int start_node,int numEdges) noexcept(false) {
 

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

    // auto Q1 = sycl::queue{devs[1]}; // [opencl:cpu:1] Intel(R) OpenCL, AMD EPYC 7742 64-Core Processor OpenCL 3.0 (Build 0) [2023.16.10.0.17_160000]
    // auto q = sycl::queue{devs[2]}; //  [ext_oneapi_cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.2]

    // std::cout << "Running on devices: " << std::endl;
    // std::cout << "1:\t" << Q1.get_device().get_info<sycl::info::device::name>()
    //           << std::endl;
    // std::cout << "2:\t" << q.get_device().get_info<sycl::info::device::name>()
              // << std::endl;
              
    // Print out the device information.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::vector<unsigned int> h_graph_pipe(numCols,0);
    h_graph_pipe[0] = start_node;
    unsigned int pipe_size=1;

    unsigned int* usm_nodes_start = malloc_device<unsigned int>(source_indptr.size(), q);
    int *usm_dist = malloc_device<int>(h_dist.size(), q); 
    MyUint1 *usm_updating_mask = malloc_device<MyUint1>(h_updating_graph_mask.size(), q); 
    MyUint1 *usm_visited= malloc_device<MyUint1>(h_graph_visited.size(), q); 
    unsigned int *usm_edges = malloc_device<unsigned int>(source_inds.size(), q); 
    unsigned int *usm_pipe = malloc_device<unsigned int>(h_graph_pipe.size(), q); 
    unsigned int *usm_pipe_size = malloc_device<unsigned int>(1, q); 
 


    initUSMvec(q,usm_edges,source_inds);
    initUSMvec(q,usm_nodes_start,source_indptr);
    initUSMvec(q,usm_dist,h_dist);
    initUSMvec(q,usm_updating_mask,h_updating_graph_mask);
    initUSMvec(q,usm_visited,h_graph_visited);
    initUSMvec(q,usm_pipe,h_graph_pipe);

std::array<event, NUM_COMPUTE_UNITS> eventsExploreRead;
std::array<double, NUM_COMPUTE_UNITS> execTimesExploreRead;

    unsigned int h_over = 1;
    unsigned int *d_over = malloc_device<unsigned int>(1, q);



    // Compute kernel execution time
    event e_explore_1,e_explore_2,e_explore_3,e_levelgen_0,e_levelgen_1,e_explore_4;
    event e_pipegen,e_maskreset,e_remove_2,e_remove_3;
    double time_kernel=0,time_kernel1=0,time_kernel2=0,time_kernel3=0,time_kernel_levelgen=0,time_kernel_levelgen_1=0,time_kernel_pipegen=0,time_kernel_maskreset=0;

  


    int global_level = 1;

    
    for(int ijk=0; ijk < 100; ijk++){
      // std::cout << h_over << " -- \n";
     if(h_over == 0){
      std::cout << "total number of iterations" << ijk << "\n";
      break;
     }    
    int zero = 0;
    q.memcpy(d_over, &zero, sizeof(unsigned int)).wait();
      
      // q.memcpy(d_over, &h_over, sizeof(unsigned int)).wait();
    // This way makes sure that there is no wait time between kernels 
    // The unrolled loop on GPU seems to have some wait overhead between first and second kernels
    // e_explore_1 = parallel_explorer_kernel<0>(q,h_over,offset[0],offset_inds[0],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
    // e_explore_2 = parallel_explorer_kernel<1>(q,h_over,offset[1],offset_inds[1],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
    // e_explore_3 = parallel_explorer_kernel<2>(q,h_over,offset[2],offset_inds[2],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
    // e_explore_4 = parallel_explorer_kernel<3>(q,h_over,offset[3],offset_inds[3],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
       fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>([&](auto krnlID) {
        eventsExploreRead[krnlID] = parallel_explorer_kernel<krnlID>(
            q,h_over,offset[krnlID],offset_inds[krnlID],usm_nodes_start,usm_edges,usm_dist,usm_pipe, usm_updating_mask,usm_visited);
      });
    q.wait();
 

    e_levelgen_0 =parallel_levelgen_kernel<0>(q,0,numCols/2,usm_dist,usm_updating_mask,usm_visited,global_level);
    e_levelgen_1 =parallel_levelgen_kernel<1>(q,numCols/2,numCols,usm_dist,usm_updating_mask,usm_visited,global_level);


    e_pipegen =pipegen_kernel<8>(q,numCols,usm_pipe, d_over,usm_updating_mask);
    q.wait();


    e_maskreset =maskremove_kernel<8>(q,numCols,usm_updating_mask);
 
// #############################################################################################              


        
         

    q.wait();
    q.memcpy(&h_over, d_over, sizeof(unsigned int)).wait();
    // h_over++;
 
//  fpga_tools::UnrolledLoop<NUM_COMPUTE_UNITS>([&](auto krnlID) {
//         execTimesExploreRead[krnlID] += GetExecutionTime(eventsExploreRead[krnlID]);
//  });
    // time_kernel  += GetExecutionTime(e_explore_1);
    // time_kernel1 += GetExecutionTime(e_explore_2);
    // time_kernel2 += GetExecutionTime(e_explore_3);
    // time_kernel3 += GetExecutionTime(e_explore_4);
    // time_kernel_levelgen += GetExecutionTime(e_levelgen_0);
    // time_kernel_levelgen_1 += GetExecutionTime(e_levelgen_1);
// time_kernel_pipegen += GetExecutionTime(e_pipegen);
// time_kernel_maskreset += GetExecutionTime(e_maskreset);
    global_level++;

    }





    // copy usm_visited back to hostArray
    q.memcpy(&h_dist[0], usm_dist, h_dist.size() * sizeof(int));

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
         "|-------------------------+-------------------------|\n",numCols,numEdges);

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