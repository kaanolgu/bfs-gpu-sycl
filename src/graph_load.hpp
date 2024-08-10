#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <stdexcept>
// CSR structure to hold the graph
struct CSRGraph {
    std::vector<Uint32> meta;	
    std::vector<Uint32> indptr;
    std::vector<Uint32> inds;

	std::vector<Uint32> metaOffsets;
    std::vector<Uint32> indptrOffsets;
    std::vector<Uint32> indsOffsets;
};

void readFromMM(const char *filename, std::vector<Uint32> &buffer) {
    std::cout << "Reading " << filename << "..." ;

    // Open the file:
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "ERROR: File " << filename << " not found!" << std::endl;
        throw std::runtime_error("File not found");
    }

    // Get file size:
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
    std::cout << "Loading matrix " << datasetName << " with " << partitionCount << " partitions.." << std::endl;
    
    std::string pth = "/dataset/";
    std::string non_switch = getenv("PWD") + pth;
    std::string temp = datasetName;
    non_switch += temp + "-csc-" + std::to_string(partitionCount) + "/" + temp + "-csc-";

    for (Uint32 i = 0; i < partitionCount; i++) {
        CSRGraph singleGraph;

        // Record the original sizes before loading new data
        graph.metaOffsets.push_back(graph.meta.size());
        graph.indptrOffsets.push_back(graph.indptr.size());
        graph.indsOffsets.push_back(graph.inds.size());


        std::string str_meta = non_switch + std::to_string(i) + "-meta.bin";
        std::string str_indptr = non_switch + std::to_string(i) + "-indptr.bin";
        std::string str_inds = non_switch + std::to_string(i) + "-inds.bin";
        
        readFromMM(str_meta.c_str(), singleGraph.meta);
        readFromMM(str_indptr.c_str(), singleGraph.indptr);
        readFromMM(str_inds.c_str(), singleGraph.inds);

        graph.meta.insert(graph.meta.end(), singleGraph.meta.begin(), singleGraph.meta.end());
        graph.indptr.insert(graph.indptr.end(), singleGraph.indptr.begin(), singleGraph.indptr.end());
        graph.inds.insert(graph.inds.end(), singleGraph.inds.begin(), singleGraph.inds.end());
    }
	// // Record the original sizes after loading new data [final size]
	// graph.metaOffsets.push_back(graph.meta.size());
	// graph.indptrOffsets.push_back(graph.indptr.size());
	// graph.indsOffsets.push_back(graph.inds.size());

    return graph;
}