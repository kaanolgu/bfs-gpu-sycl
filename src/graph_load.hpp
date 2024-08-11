#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <stdexcept>
// CSR structure to hold the graph
struct CSRGraph {
    // Use single vectors for when partitionCount is 1
    std::vector<Uint32> meta;
    std::vector<Uint32> indptr;
    std::vector<Uint32> inds;

    // Use vectors of vectors for when partitionCount is greater than 1
    std::vector<std::vector<Uint32>> metaMulti;
    std::vector<std::vector<Uint32>> indptrMulti;
    std::vector<std::vector<Uint32>> indsMulti;

    // Flag to check if we're using multi-dimensional vectors
    bool isMultiDimensional = false;
};

void readFromMM(const char *filename, std::vector<Uint32> &buffer) {
    std::cout << "Reading " << filename << "..." ;

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
    buffer.resize(originalSize + fileSize / sizeof(Uint32));

    file.read(reinterpret_cast<char *>(buffer.data() + originalSize), fileSize);

    std::cout << " OK" << std::endl;
}

CSRGraph loadMatrix(Uint32 partitionCount, std::string datasetName) {
    CSRGraph graph;
    std::cout << "Loading matrix " << datasetName << " with " << partitionCount << " partitions..." << std::endl;
    
    std::string pth = "/dataset/";
    std::string non_switch = getenv("PWD") + pth;
    std::string temp = datasetName;
    non_switch += temp + "-csc-" + std::to_string(partitionCount) + "/" + temp + "-csc-";

    if (partitionCount == 1) {
        // Load data into single-dimensional vectors
        std::string str_meta = non_switch + "0-meta.bin";
        std::string str_indptr = non_switch + "0-indptr.bin";
        std::string str_inds = non_switch + "0-inds.bin";
        
        readFromMM(str_meta.c_str(), graph.meta);
        readFromMM(str_indptr.c_str(), graph.indptr);
        readFromMM(str_inds.c_str(), graph.inds);
    } else {
        // Use multi-dimensional vectors for multiple partitions
        graph.isMultiDimensional = true;
        graph.metaMulti.resize(partitionCount);
        graph.indptrMulti.resize(partitionCount);
        graph.indsMulti.resize(partitionCount);

        for (Uint32 i = 0; i < partitionCount; i++) {
            std::string str_meta = non_switch + std::to_string(i) + "-meta.bin";
            std::string str_indptr = non_switch + std::to_string(i) + "-indptr.bin";
            std::string str_inds = non_switch + std::to_string(i) + "-inds.bin";
            
            readFromMM(str_meta.c_str(), graph.metaMulti[i]);
            readFromMM(str_indptr.c_str(), graph.indptrMulti[i]);
            readFromMM(str_inds.c_str(), graph.indsMulti[i]);
        }
    }

    return graph;
}