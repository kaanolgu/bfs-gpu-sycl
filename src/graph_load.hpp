#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <stdexcept>
// CSR structure to hold the graph
struct CSRGraph {
    // Use single vectors for when partitionCount is 1
    std::vector<uint32_t> meta32;
    std::vector<uint64_t> meta64;
    std::vector<uint32_t> indptr;
    std::vector<uint32_t> inds;

    // Use vectors of vectors for when partitionCount is greater than 1
    std::vector<std::vector<uint32_t>> indptrMulti;
    std::vector<std::vector<uint32_t>> indsMulti;

    // Flag to check if we're using multi-dimensional vectors
    bool isMultiDimensional = false;
};


void readFromMM(const char *filename, std::vector<uint32_t> &buffer) {
    #if VERBOSE == 1
    std::cout << "- Reading " << filename << "..." ;
    #endif
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "ERROR: File " << filename << " not found!" << std::endl;
        throw std::runtime_error("File not found");
    }

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize == 0) {
        std::cerr << "ERROR: File size is 0!" << std::endl;
        throw std::runtime_error("File size is 0");
    }

    size_t originalSize = buffer.size();
    buffer.resize(originalSize + fileSize / sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(buffer.data() + originalSize), fileSize);
#if VERBOSE == 1
    std::cout << " OK" << std::endl;
    #endif
}
void readFromMM_meta(const char *filename, std::vector<uint32_t> &buffer32, std::vector<uint64_t> &buffer64) {
    #if VERBOSE == 1
    std::cout << "- Reading " << filename << "..." ;
    #endif
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "ERROR: File " << filename << " not found!" << std::endl;
        throw std::runtime_error("File not found");
    }

    // Reading data in the specified pattern
    uint32_t data32;
    uint64_t data64;
    while (file) {
        // Read the first 4-byte integer
        file.read(reinterpret_cast<char*>(&data32), sizeof(uint32_t));
        if (!file) break; // Check for EOF
        buffer32.push_back(data32);

        // Read the second 4-byte integer
        file.read(reinterpret_cast<char*>(&data32), sizeof(uint32_t));
        if (!file) break;
        buffer32.push_back(data32);

        // Read the 8-byte integer
        file.read(reinterpret_cast<char*>(&data64), sizeof(uint64_t));
        if (!file) break;
        buffer64.push_back(data64);

        // Read the final 4-byte integer
        file.read(reinterpret_cast<char*>(&data32), sizeof(uint32_t));
        if (!file) break;
        buffer32.push_back(data32);
    }

    if (file.bad()) {
        std::cerr << "ERROR: Error reading file " << filename << std::endl;
        throw std::runtime_error("Error reading file");
    }

#if VERBOSE == 1
    std::cout << " OK" << std::endl;
    #endif
}
CSRGraph loadMatrix(uint32_t partitionCount, std::string datasetName) {
    CSRGraph graph;
    #if VERBOSE == 1
    std::cout << "Loading matrix " << datasetName << " with " << partitionCount << " partitions..." << std::endl;
    #endif
    std::string pth = "/dataset/";
    std::string non_switch = getenv("PWD") + pth;
    std::string temp = datasetName;
    non_switch += temp + "-csc-" + std::to_string(partitionCount) + "/" + temp + "-csc-";
    std::string str_meta = non_switch + "meta.bin";
    readFromMM_meta(str_meta.c_str(), graph.meta32,graph.meta64);
    
    if (partitionCount == 1) {
        // Load data into single-dimensional vectors
        std::string str_indptr = non_switch + "0-indptr.bin";
        std::string str_inds = non_switch + "0-inds.bin";
        
        
        readFromMM(str_indptr.c_str(), graph.indptr);
        readFromMM(str_inds.c_str(), graph.inds);
    } else {
        // Use multi-dimensional vectors for multiple partitions
        graph.isMultiDimensional = true;

        graph.indptrMulti.resize(partitionCount);
        graph.indsMulti.resize(partitionCount);

        for (uint32_t i = 0; i < partitionCount; i++) {
            // std::string str_meta = non_switch + std::to_string(i) + "-meta.bin";
            std::string str_indptr = non_switch + std::to_string(i) + "-indptr.bin";
            std::string str_inds = non_switch + std::to_string(i) + "-inds.bin";
            

            readFromMM(str_indptr.c_str(), graph.indptrMulti[i]);
            readFromMM(str_inds.c_str(), graph.indsMulti[i]);
        }
    }

    return graph;
}