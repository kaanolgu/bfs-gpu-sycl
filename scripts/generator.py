#!/usr/bin/env python


import io, sys, numpy, scipy, struct, os
from scipy import io
from scipy import sparse
from numpy import inf
import numpy as np
import networkx as nx

def round8(a):
    return int(a) + 4 & ~7

num_cu = [1,2,3,4,5,6,7,8]

absolute_path = os.path.dirname(__file__)
relative_path = "../dataset/"
relative_path_localtxt = "txt/"
graphDataRoot = os.path.join(absolute_path, relative_path)
localtxtFolder = os.path.join(absolute_path, relative_path_localtxt)

def nnzSplit(
    matrix: sparse.sparray, n_compute_units: int = 4,
) -> list[sparse.sparray]:
    nnz = matrix.getnnz(axis=1).cumsum()

    total = nnz[-1]
    ideal_breaks = np.arange(0, total, total/n_compute_units)
    break_idx = [*nnz.searchsorted(ideal_breaks),matrix.shape[0]]
    # make sure that break_idx is divisible by MAXIMUM_NUM_GPUs per node
    break_idx = [round8(x) for x in break_idx]
    # return [
    #     matrix[i: j,:]
    #     for i, j in zip(break_idx[:-1], break_idx[1:])
    # ]
    partitions = [
        matrix[i: j, :].astype(np.uint32)  # Ensures that the partitions are of type uint32
        for i, j in zip(break_idx[:-1], break_idx[1:])
    ]
    return partitions

def rowSplit(
    matrix: sparse.sparray, n_compute_units: int = 4,
) -> list[sparse.sparray]:
    total = matrix.shape[0]
    stepSize = int(alignedIncrement(total / n_compute_units,0,64))
    break_idx = np.arange(0, total, stepSize)
    break_idx = np.append(break_idx,total)
    # make sure that break_idx is divisible by MAXIMUM_NUM_GPUs
    break_idx = [round8(x) for x in break_idx]
    return [
        matrix[i: j,:]
        for i, j in zip(break_idx[:-1], break_idx[1:])
    ]


def makeGraphList():
    graphs = []
    # Get all the files ending with .mat extension 
    for file in os.listdir(localtxtFolder):
    	if file.endswith(".txt"):
        	graphs += [file.rsplit( ".", 1 )[0]]
    # print(graphs)
    print("# of Found graphs in directory :",len(graphs))
    # graphs = ["rmat-19-32"]
    return graphs

def buildGraphManager(dim,pick,csr = False):
    graphs = makeGraphList()
    for g in graphs:
        m = GraphMatrix()
        graph = loadGraph(g,dim)
        print("PATH : ", localtxtFolder + g + ".txt")
        if (csr):
            g += "-" + pick + "-csr"
        else:
            g += "-" + pick + "-csc"
            # SpMV BFS needs transpose of matrix
            graph = graph.transpose()
        g = g.replace("/", "-")
        for num_partition in num_cu:
            print("Graph " + g + " with " + str(num_partition) + " partitions")
            m.prepareGraph(g, num_partition, graph,csr, pick)
        
      
def buildGraphManagerSingle(name,dim,pick, csr = False):
    g = name
    m = GraphMatrix()

    # Load the graph once outside the loop
    graph = loadGraph(g,dim)
    print("PATH : ", localtxtFolder + g + ".txt")
    if (csr):
        g += "-" + pick + "-csr"
    else:
        g += "-" + pick + "-csc"
        # SpMV BFS needs transpose of matrix
        graph = graph.transpose()
    g = g.replace("/", "-")
    # Call prepareGraph with the loaded graph for each partition count
    for num_partition in num_cu:
        print("Graph " + g + " with " + str(num_partition) + " partitions")
        m.prepareGraph(g, num_partition, graph,csr, pick)
      


class GraphMatrix:
    def __init__(self):
        self.copyCommandBuffer = []
        self.graphName = ""

    def resetCommandBuffer(self):
        self.copyCommandBuffer = []

    def serializeGraphData(self, graph, name, rootFolder,index,PrevRowsValue):
        # Check that the data type is correct
        print(f"graph.indices dtype  {graph.indices.dtype}")
        print(f"graph.indptr dtype  {graph.indptr.dtype}")

        # save index pointers
        fileName = rootFolder + "/" + name + "-indptr.bin"
        indPtrFile = open(fileName, "wb")
        indPtrFile.write(graph.indptr)
        indPtrFile.close()

        # save indices
        fileName = rootFolder + "/" + name + "-inds.bin"
        indsFile = open(fileName, "wb")
        graph.indices = graph.indices + PrevRowsValue
        indsFile.write(graph.indices)
        indsFile.close()

        print("Rows = " + str(graph.shape[0]))
        print("Cols = " + str(graph.shape[1]))
        print("NonZ = " + str(graph.nnz))

        # return the xmd command and new start address
        return graph.shape[0]



    def prepareGraph(self, graphName, partitionCount, graph, csr, pick):
        # create the graph partitions list
        partitions = []
        if(pick == "row"):
            partitions = rowSplit(graph,partitionCount)
        elif pick == "nnz":
            partitions = nnzSplit(graph,partitionCount)
        # add subfolder with name and part count
        targetDir = graphDataRoot + graphName + "-" + str(partitionCount)
        # create the target dir if it does not exist
        if not os.path.exists(targetDir):
            os.makedirs(targetDir)
        # serialize the graph data and build commands
        i = 0

        startRow = 0
        savedRows = 0
        metafileName = targetDir + "/" + graphName + "-meta.bin"
        metaDataFile = open(metafileName, "wb")

        for i, part in enumerate(partitions):
            # savedRows = break_idx[i]
            print ("\n------------\n"+"Partition " + str(i) + "\n------------")
            # write the metadata base ptr into
            if (csr):
                res = self.serializeGraphData(part, graphName + "-" + str(i),
                                              targetDir,i,savedRows)
            else:
                res = self.serializeGraphData(part, graphName + "-" + str(i),
                                              targetDir,i,savedRows)
            savedRows += res
            metaDataFile.write(struct.pack("I", part.shape[0]))
            metaDataFile.write(struct.pack("I", part.shape[1]))
            metaDataFile.write(struct.pack("I", part.nnz))
            metaDataFile.write(struct.pack("I", startRow))
            #update start row
            startRow += part.shape[0]  
        
        metaDataFile.close()
        print ("Graph " + graphName + " prepared with " + str(partitionCount) + " partitions")
        if csr:
            print("Matrix stored in row-major format")
        else:
            print ("Matrix stored in col-major format")
        print ("All data is located in " + targetDir + "\n")

# Function to count comment lines
def count_comment_lines(filepath, comment_chars=['%', '%%']):
    with open(filepath, 'r') as file:
        # Initialize line count
        comment_line_count = 0
        for line in file:
            # Strip leading/trailing whitespace and check if the line starts with any of the comment characters
            if line.strip().startswith(tuple(comment_chars)):
                comment_line_count += 1
            else:
                # Stop counting when the first non-comment line is found
                break
    return comment_line_count

def loadGraph(matrix,dim):
    name_matrix = str(matrix) + '.txt'
    path_to_go = localtxtFolder + name_matrix
    
    # load matrix from local file system
    # r = scipy.io.loadmat(path_to_go)['M']
    # List of target filenames with mtx format
    file_names_from_mtx = [
        "webbase-1M.txt",
        "kron_g500-logn21.txt",
        "indochina-2004.txt",
        "hollywood-2009.txt",
        "europe_osm.txt"
    ]

    # Check if path_to_go matches any of the filenames
    if name_matrix in file_names_from_mtx:
        print(f"{path_to_go} is in the list of target filenames.")
        # Count the number of comment lines
        num_comment_lines = count_comment_lines(path_to_go)

        # Load the data using numpy.loadtxt
        arr = np.loadtxt(path_to_go, dtype=np.uint32, comments='%', skiprows=num_comment_lines + 1)
    else:
        arr = np.loadtxt(path_to_go, dtype=np.uint32,comments=['#', '$'])

    data = np.ones((len(arr[:, 0]),), dtype=np.uint32)
    row = arr[:, 0]
    col = arr[:, 1]
    csr_matrix = sparse.csr_matrix((data, (row, col)), shape=(dim, dim))


    return csr_matrix


# increment base address by <increment> and ensure alignment to <align>
def alignedIncrement(base, increment, align):
    res = base + increment
    rem = res % align
    if rem != 0:
        res += align - rem
    return res

if __name__ == '__main__':
    if(sys.argv[1] == "all"):
        partition_mode = sys.argv[2] ## nnz or row
        buildGraphManager(dim,partition_mode)
    else:
        dataset_name = sys.argv[1]
        partition_mode = sys.argv[2] ## nnz or row
        dim = int(sys.argv[3]) # number of nodes
        buildGraphManagerSingle(dataset_name,dim,partition_mode)