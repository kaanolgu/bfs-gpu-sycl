#pragma once
// graph_load.hpp — HDF5 edition
//
// The HDF5 file layout expected:
//
//   <graph>-<mode>-csc.h5          (one file per dataset)
//     gpus_<N>/
//       meta32    [uint32 dataset]
//       meta64    [uint64 dataset]
//       partition_0/
//         indptr   [uint32 dataset]
//         indices  [uint32 dataset]
//       partition_1/ ...
//       ...

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <thread>
#include <H5Cpp.h>


struct CSRGraph {
    std::vector<uint32_t> meta32;
    std::vector<uint64_t> meta64;

    // Single-partition path
    std::vector<uint32_t> indptr;
    std::vector<uint32_t> inds;

    // Multi-partition path
    std::vector<std::vector<uint32_t>> indptrMulti;
    std::vector<std::vector<uint32_t>> indsMulti;

    bool isMultiDimensional = false;
};



namespace hdf5_detail {

/// Read a 1-D HDF5 dataset into a std::vector<T>.
template <typename T>
inline void readDataset(const H5::Group &group,
                        const std::string &name,
                        std::vector<T> &dst,
                        const H5::PredType &memType) {
    H5::DataSet ds = group.openDataSet(name);
    H5::DataSpace space = ds.getSpace();
    hsize_t n = space.getSimpleExtentNpoints();
    dst.resize(static_cast<size_t>(n));
    ds.read(dst.data(), memType);
#if VERBOSE == 1
    std::cout << "  - read " << name << "  (" << n << " elements)" << std::endl;
#endif
}

} // namespace hdf5_detail

// Main loader

inline CSRGraph loadMatrix(uint32_t partitionCount,
                           const std::string &datasetName) {
    CSRGraph graph;

    // Build path to the single HDF5 file for this dataset
    std::string pth  = "/dataset/";
    std::string base = std::string(getenv("PWD")) + pth;
    std::string h5path = base + datasetName + "-csc.h5";

#if VERBOSE == 1
    std::cout << "Loading " << h5path
              << "  (gpus_" << partitionCount << ")" << std::endl;
#endif

    H5::H5File file(h5path, H5F_ACC_RDONLY);

    // Open the group for the requested partition count
    std::string groupName = "gpus_" + std::to_string(partitionCount);
    H5::Group partGroup = file.openGroup(groupName);

    // Meta arrays
    hdf5_detail::readDataset(partGroup, "meta32", graph.meta32,
                        H5::PredType::NATIVE_UINT32);
    hdf5_detail::readDataset(partGroup, "meta64", graph.meta64,
                        H5::PredType::NATIVE_UINT64);

    // Partition data 
    if (partitionCount == 1) {
        H5::Group p = partGroup.openGroup("partition_0");
        hdf5_detail::readDataset(p, "indptr",  graph.indptr,
                            H5::PredType::NATIVE_UINT32);
        hdf5_detail::readDataset(p, "indices", graph.inds,
                            H5::PredType::NATIVE_UINT32);
    } else {
        graph.isMultiDimensional = true;
        graph.indptrMulti.resize(partitionCount);
        graph.indsMulti.resize(partitionCount);

        std::vector<std::thread> threads;
        for (uint32_t i = 0; i < partitionCount; ++i) {
            threads.emplace_back([&, i]() {
                // Each thread opens its own file handle (HDF5 is not thread-safe on shared handles)
                H5::H5File localFile(h5path, H5F_ACC_RDONLY);
                H5::Group localGroup = localFile.openGroup(groupName + "/partition_" + std::to_string(i));
                hdf5_detail::readDataset(localGroup, "indptr",  graph.indptrMulti[i],
                                        H5::PredType::NATIVE_UINT32);
                hdf5_detail::readDataset(localGroup, "indices", graph.indsMulti[i],
                                        H5::PredType::NATIVE_UINT32);
            });
        }
        for (auto &t : threads) t.join();
    }
  }

  return graph;
}