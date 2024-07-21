#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;
using Uint32 = unsigned int;
template <typename T>
int upper_bound(T* arr, int length, T value) {
    int left = 0;
    int right = length;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (value < arr[mid]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}
int main() {
    auto device_selector = sycl::gpu_selector_v;
     sycl::queue q{device_selector,
                sycl::property::queue::enable_profiling{}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;


     // Example data: degrees of vertices for each thread
    std::vector<Uint32> degrees = {2, 3, 5, 1}; // Sample degree array
    Uint32 length = degrees.size();

    // Allocate device memory for degrees array
    Uint32* d_degrees = malloc_shared<Uint32>(length, q);
    for (int i = 0; i < length; ++i) {
        d_degrees[i] = degrees[i];
    }

     
     Uint32 *a_degrees = malloc_shared<Uint32>(length, q);
    // Allocate USM memory for the aggregate result
    int* aggregate_degree_per_block = malloc_shared<int>(1, q);
    *aggregate_degree_per_block = 0;
    nd_range<1> range(length, 4);
    // Perform exclusive prefix sum on the device
    auto e = q.submit([&](handler& h) {
        // Local memory for exclusive sum, shared within the work-group
        local_accessor<int, 1> local_sum(length, h);

        h.parallel_for<class LevelGenerator>(range, [=](nd_item<1> item) [[intel::kernel_args_restrict]] {
            int localIdx = item.get_local_id(0);
          const int blockDim  = item.get_local_range(0); // blockDim.x

            // Initialize thread-local th_deg
            int th_deg = d_degrees[localIdx];
            local_sum[localIdx] = th_deg;
            item.barrier(access::fence_space::local_space);

            // Perform the scan operation in local memory
            for (int offset = 1; offset < length; offset *= 2) {
                int temp = (localIdx >= offset) ? local_sum[localIdx - offset] : 0;
                item.barrier(access::fence_space::local_space);
                local_sum[localIdx] += temp;
                item.barrier(access::fence_space::local_space);
            }

            // Write the exclusive scan result back to the thread-local th_deg
            if (localIdx > 0) {
                th_deg = local_sum[localIdx - 1];
            } else {
                th_deg = 0;
            }

            // Calculate aggregate degree per block
            if (localIdx == length - 1) {
                *aggregate_degree_per_block = local_sum[length - 1];
            }

            item.barrier(sycl::access::fence_space::local_space);




           // Store back to shared memory (to later use in the binary search).
          a_degrees[localIdx] = th_deg;
            item.barrier(sycl::access::fence_space::local_space);

            for (unsigned int i = localIdx;            // threadIdx.x
       i < *aggregate_degree_per_block;  // total degree to process
       i += blockDim     // increment by blockDim.x
  ) {
     auto it = upper_bound(d_degrees,length, i);
      // int id = std::distance(degrees, it) - 1;
    unsigned int id = it-1; // Return the distance minus 1
    if (id < length){
    auto e = 33 + i - d_degrees[id];
            // Debug: Print the thread-local th_deg for verification
            printf("i: %d, it = %d, id = %d, e = %d\n", i, it,id,e);
  }
        }
        });
    });
   q.wait();
    // Output the aggregate degree per block
    std::cout << "Aggregate degree per block: " << *aggregate_degree_per_block << std::endl;
    for(int i =0; i < length ; i++)
    std::cout << "degrees["<< i << "]: " << degrees[i] << std::endl;
    // Free USM memory
    free(aggregate_degree_per_block, q);

    return 0;
}
