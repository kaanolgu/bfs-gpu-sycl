using namespace sycl;
 using namespace std::chrono;

#include <math.h>
#include <iostream>
#include <vector>
// #include <sycl/ext/intel/fpga_extensions.hpp>
#include <bitset>
#include "functions.hpp"

// Intel Compatibility Tool (aka c2s)
// #include <dpct/dpct.hpp>
// #include <dpct/rng_utils.hpp>
// #include <dpct/dpl_utils.hpp>
#include <sycl/sycl.hpp>
// This function returns a vector of two (not necessarily distinct) devices,
// allowing computation to be split across said devices.
#define NUMBER_OF_WORKGROUPS 256

// Aliases for LSU Control Extension types
// Implemented using template arguments such as prefetch & burst_coalesce
// on the new ext::intel::lsu class to specify LSU style and modifiers
// using PrefetchingLSU = ext::intel::lsu<ext::intel::prefetch<true>,ext::intel::statically_coalesce<false>>;
// using PipelinedLSU = ext::intel::lsu<>;
// using BurstCoalescedLSU = ext::intel::lsu<ext::intel::burst_coalesce<true>,ext::intel::statically_coalesce<false>>;
// using CacheLSU = ext::intel::lsu<ext::intel::burst_coalesce<true>, ext::intel::cache<1024*1024>,ext::intel::statically_coalesce<false>>;
template <typename vertex_t>
class Limits {
public:
    static vertex_t invalid() {
        return std::numeric_limits<vertex_t>::max();
    }

    static bool is_valid(vertex_t v) {
        return v != invalid();
    }
};

// Define a custom binary operation
struct plus_op {
    template <typename T>
    T operator()(T a, T b) const {
        return a + b;
    }
};


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

// bool search(
//                       Uint32 const& source,    // ... source
//                       Uint32 const& neighbor,  // neighbor
//                       Uint32 const& edge        // edge
//                       ){
//       // If the neighbor is not visited, update the distance. Returning false
//       // here means that the neighbor is not added to the output frontier, and
//       // instead an invalid vertex is added in its place. These invalides (-1 in
//       // most cases) can be removed using a filter operator or uniquify.

//       // if (distances[neighbor] != std::numeric_limits<vertex_t>::max())
//       //   return false;
//       // else
//       //   return (math::atomic::cas(
//       //               &distances[neighbor],
//       //               std::numeric_limits<vertex_t>::max(), iteration + 1) ==
//       //               std::numeric_limits<vertex_t>::max());

//       // Simpler logic for the above.
//       auto old_distance = sycl::atomic_ref<std::int32_t, 
//                                memory_order::relaxed, 
//                                memory_scope::device, 
//                                access::address_space::global_space>(distances[neighbor])
//                         .fetch_min(iteration + 1);

// return (iteration + 1 < old_distance);
//     };

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
                                int N,
                                int V, // number of vertices
                                Uint32 iteration,
                                Uint32* Offset,
                                Uint32 *Edges,
                                Uint32* Frontier,
                                Uint32 *frontierCount,
                                MyUint1 *VisitMask,
                                Uint32 *Distance,
                                Uint32 *PrefixSum)
    {

   // Define the work-group size and the number of work-groups
    const size_t local = NUMBER_OF_WORKGROUPS;  // Number of work-items per work-group
    const size_t global = ((V + local - 1) / local) * local;
    // int ncus = q.get_device().get_info<info::device::max_compute_units>();

    auto search = [Distance, iteration](
                      Uint32 const& source,    // ... source
                      Uint32 const& neighbor,  // neighbor
                      Uint32 const& edge        // edge
                      ) -> bool {
      // If the neighbor is not visited, update the distance. Returning false
      // here means that the neighbor is not added to the output frontier, and
      // instead an invalid vertex is added in its place. These invalides (-1 in
      // most cases) can be removed using a filter operator or uniquify.

      // if (distances[neighbor] != std::numeric_limits<vertex_t>::max())
      //   return false;
      // else
      //   return (math::atomic::cas(
      //               &distances[neighbor],
      //               std::numeric_limits<vertex_t>::max(), iteration + 1) ==
      //               std::numeric_limits<vertex_t>::max());

      // Simpler logic for the above.
      // auto old_distance =
      //     math::atomic::min(&Distance[neighbor], iteration + 1);

      auto old_distance = sycl::atomic_ref<Uint32, 
                               memory_order::relaxed, 
                               memory_scope::device, 
                               access::address_space::global_space>(Distance[neighbor])
                        .fetch_min(iteration + 1);
      return (iteration + 1 < old_distance);
    };








    // std::cout<< "number of maximumx compute units" << ncus << "\n";
    // Allocate USM shared memory
    // Uint32 *degrees = malloc_device<Uint32>(256, q);
    // Uint32 *sedges = malloc_device<Uint32>(256, q);
    // Uint32 *vertices = malloc_device<Uint32>(256, q);
    // std::vector<Uint32> vertices_a(256,0);
    Uint32 *block_offsets = malloc_device<Uint32>(1, q);
    unsigned long *OOffset = malloc_device<unsigned long>(1, q);



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


int* _aggregate_degree_per_block = malloc_shared<int>(local, q);
    *_aggregate_degree_per_block = 0;

int* _th_deg = malloc_shared<int>(1, q);
    *_th_deg = 0;

    // Setup the range
    nd_range<1> range(global, local);

        auto e = q.submit([&](handler& h) {
          // Local memory for exclusive sum, shared within the work-group
          sycl::local_accessor<int> local_sum(local, h);
          sycl::local_accessor<Uint32> vertices(local, h);
          sycl::local_accessor<Uint32> sedges(local, h);
          sycl::local_accessor<Uint32> degrees(local, h);
        h.parallel_for<class ExploreNeighbours>(range, [=](nd_item<1> item) {
          const int gid = item.get_global_id(0);    // global id
          const int lid  = item.get_local_id(0); // threadIdx.x
          const int blockIdx  = item.get_group(0); // blockIdx.x
          const int gridDim   = item.get_group_range(0); // gridDim.x
          const int blockDim  = item.get_local_range(0); // blockDim.x
          Uint32 th_deg[ITEMS_PER_THREAD]; // this variable is shared between workitems
        // this variable will be instantiated for each work-item separately

     

        device_ptr<Uint32> DevicePtr_start(Offset);  
        device_ptr<Uint32> DevicePtr_end(Offset + 1);  
        device_ptr<Uint32> DevicePtr_edges(Edges); 
        // device_ptr<MyUint1> DevicePtr_visited(Visit);  

            if (gid < V) {
                Uint32 v = Frontier[gid];
                vertices[lid] = v;
                
                if (Limits<Uint32>::is_valid(v)) {
                    sedges[lid] = DevicePtr_start[v];
                    th_deg[0] =DevicePtr_end[v] - DevicePtr_start[v];
                } else {
                    th_deg[0] = 0;
                }
            }
            else {
                vertices[lid] = Limits<Uint32>::invalid();
                th_deg[0] = 0;
            }
        
            if(blockDim * blockIdx + lid == 0)
            printf("lid: %d, vertices[lid]: %d, th_deg[0]: %d\n",lid,vertices[lid],th_deg[0]);
     
            

            sycl::group_barrier(item.get_group());
        //  item.barrier(sycl::access::fence_space::local_space);
  //  });
  //       });
  //       copyToHost(q,vertices,vertices_a);
  //       for(int i : vertices_a){
  //         std::cout << i << std::endl;
  //       }
        // DEBUG
        // ––––––––
        //  for(int j = 0; j < degrees[lid]; j++) {
        //     int iterator =  sedges[lid] + j;
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
 
         
            

            // item.barrier(access::fence_space::local_space);
 
            // group_barrier(item.get_group());

            // // Perform the scan operation in local memory
            // for (int d = 1; d < log2(local); d++) {
            //     Uint32 stride = (1 << d);
            //     int update = (lid >= stride) ? local_sum[lid - stride] : 0;
            //     // group_barrier(item.get_group());
            //     item.barrier(access::fence_space::local_space);
            //     local_sum[lid] += update;
            //     // group_barrier(item.get_group());
            //     item.barrier(access::fence_space::local_space);
            // }

            // // Write the exclusive scan result back to the thread-local th_deg
            // if (lid > 0) {
            //     th_deg[0] = local_sum[blockDim - 1];
            // } else {
            //     th_deg[0] = 0;
            // }

            // // Calculate aggregate degree per block
            // if (lid == blockDim - 1) {
            //     aggregate_degree_per_block = local_sum[lid];
            // }
      
            unsigned int exclusive_sum = sycl::exclusive_scan_over_group(item.get_group(), th_deg[0], sycl::plus<>());
            // this generates
            // lid 0, agg =0
            // lid 1, agg = 20203
            // printf("global-id: %d, agg: %d\n",gid,aggregate_degree_per_block);
            local_sum[lid] = (lid ==0) ? th_deg[0] : exclusive_sum;
                      item.barrier(access::fence_space::local_space);

            unsigned int aggregate_degree_per_block = local_sum[lid];

            // printf("+ lid = %d, agg = %d\n", lid, aggregate_degree_per_block);
            //  *(_aggregate_degree_per_block) = aggregate_degree_per_block;
    
           // Store back to shared memory (to later use in the binary search).
          degrees[lid] = exclusive_sum;
         
        //         for (int j = 0; j < th_deg[0]; j++) {
        //           int iterator =  sedges[lid] + j - degrees[lid];
        //     int id = DevicePtr_edges[iterator];
        //     MyUint1 visited_condition = DevicePtr_visited[id];
        //     if (!visited_condition) {
        //         VisitMask[id]=1;

        //     }
        //  }

        // }

        /// 3. Compute block offsets if there's an output frontier.
          // group_barrier(item.get_group());
          // if(lid == 0){
          //   sycl::atomic_ref<Uint32, sycl::memory_order::relaxed, 
          //                            sycl::memory_scope::device, 
          //                            sycl::access::address_space::global_space> atomic_ref(block_offsets[0]);
          //           OOffset[0] = atomic_ref.fetch_add(aggregate_degree_per_block);
          // }
     
       
          item.barrier(access::fence_space::local_space);


          auto length = gid - lid + blockDim;
          if (V < length)
            length = V;
          length -= gid - lid;

      // printf("blockDim = %d", blockDim);
      // int agg_cnt =0;
      // if(aggregate_degree_per_block > 0)
      // printf("agg = %d\n", aggregate_degree_per_block);
      
      //  if(gid == 0)
       printf("lid = %d, sedges[lid] = %d, degrees[lid] = %d, agg = %d\n", lid,sedges[lid], degrees[lid], aggregate_degree_per_block);
      
  for (int i = lid;            // threadIdx.x
       i < aggregate_degree_per_block;  // total degree to process
       i += blockDim    // increment by blockDim.x
  ) {

  /// 4. Compute. Using binary search, find the source vertex each thread is
  /// processing, and the corresponding edge, neighbor and weight tuple. Passed
  /// to the user-defined lambda operator to process. If there's an output, the
  /// resultant neighbor or invalid vertex is written to the output frontier.
    // Implement a simple upper_bound algorithm for use in SYCL
  
     auto it = upper_bound(degrees,length, i);
      // int id = std::distance(degrees, it) - 1;
    unsigned int id = it-1; // Return the distance minus 1
    // if (id < length){
    
    Uint32 v = vertices[id],e,n;              // source
    if (Limits<Uint32>::is_valid(v)){
    // Read from the frontier
      e = sedges[id] + i  - degrees[id]; 
      n  = DevicePtr_edges[e];   

    bool cond =search(v,n,e);
// Debug: Print the thread-local th_deg for verification
              
      if (cond) {
      VisitMask[n] = 1;
      }
    }
      if(gid == 0)
    printf("i: %d, it = %d, l=%d id = %d, v = %d, e = %d, n = %d, offset[0] = %lu\n", i, it,length,id,v,e,n,OOffset[0]);
    // if(blockDim * blockIdx + lid == 1)
    // printf("i: %d, it = %d, l=%d id = %d, v = %d, e = %d, n = %d, offset[0] = %lu\n", i, it,length,id,v,e,n,OOffset[0]);
}
        
        
    });
        });
q.wait();
// Output the aggregate degree per block
    std::cout << "Aggregate degree per block: " << *_aggregate_degree_per_block << std::endl;
    std::cout << "Aggregate +1 per block: " << *(_aggregate_degree_per_block +1) << std::endl;
    std::cout << "Aggregate +2 per block: " << *(_aggregate_degree_per_block +2)<< std::endl;
    std::cout << "Aggregate +3 per block: " << *(_aggregate_degree_per_block +3)<< std::endl;
    std::cout << "th_deg: " << *_th_deg << std::endl;
    int agg_sum = 0;
    for(int i = 0; i < local ; i++){
      agg_sum += *(_aggregate_degree_per_block + i);
    }
     std::cout << "Total agg: " <<  agg_sum<< std::endl;
    
return e;
}



event parallel_levelgen_kernel(queue &q,
                                int V,
                                Uint32 *Distance,
                                MyUint1 *VisitMask,
                                MyUint1 *Visit,
                                int iteration
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
              // Distance[gid] = iteration;
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
                  std::vector<Uint32> &DistanceHost,
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
    Uint32 *DistanceDevice    = malloc_device<Uint32>(DistanceHost.size(), q); 
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



    int iteration = 0;
    int zero = 0;

    for(int ijk=0; ijk < 100; ijk++){
      if(frontierCountHost[0] == 0){
        std::cout << "total number of iterations" << ijk << "\n";
        break;
      }    
      q.memcpy(frontierCountDevice, &zero, sizeof(Uint32)).wait();  
      exploreEvent = parallel_explorer_kernel<1>(q,vertexCount,frontierCountHost[0],iteration,OffsetDevice,EdgesDevice,FrontierDevice,frontierCountDevice, VisitMaskDevice,DistanceDevice,PrefixSumDevice);
      q.wait();
      // Level Generate
      levelEvent =parallel_levelgen_kernel(q,vertexCount,DistanceDevice,VisitMaskDevice,VisitDevice,iteration);
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
      iteration++;
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
         "|-------------------------+-------------------------|\n",vertexCount,edgeCount,(iteration-1));

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