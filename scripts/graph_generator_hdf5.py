#!/usr/bin/env python


import io, sys, numpy, scipy, struct, os
from scipy import io
from scipy import sparse
from numpy import inf
import numpy as np
import h5py

def round8(a):
    return int(a) + 4 & ~7

num_cu = [1,2,3,4,5,6,7,8]

absolute_path = os.path.dirname(__file__)
relative_path = "../dataset/"
relative_path_localtxt = "txt/"
graphDataRoot = os.path.join(absolute_path, relative_path)
localtxtFolder = os.path.join(absolute_path, relative_path_localtxt)

# HDF5 compression settings
COMPRESSION = "gzip"
COMPRESSION_OPTS = 4

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
        m.prepareGraph(g, graph, csr, pick)
        
      
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
    m.prepareGraph(g, graph, csr, pick)
      


class GraphMatrix:
    def __init__(self):
        self.copyCommandBuffer = []
        self.graphName = ""

    def resetCommandBuffer(self):
        self.copyCommandBuffer = []

    def serializeGraphData(self, gpu_group, graph, name, index, PrevRowsValue):
        # Check that the data type is correct
        print(f"graph.indices dtype  {graph.indices.dtype}")
        print(f"graph.indptr dtype  {graph.indptr.dtype}")

        # Convert `indptr` to uint32 if it's not already
        indptr_data = graph.indptr
        if indptr_data.dtype != np.uint32:
            if graph.nnz <= np.iinfo(np.uint32).max:
                indptr_data = indptr_data.astype(np.uint32)

        indices_data =  graph.indices + PrevRowsValue
        if indices_data.dtype != np.uint32:
            if graph.nnz <= np.iinfo(np.uint32).max:
                indices_data = indices_data.astype(np.uint32)

        # save into HDF5 partition group
        pgrp = gpu_group.create_group(f"partition_{index}")
        pgrp.create_dataset("indptr", data=indptr_data,
                            compression=COMPRESSION, compression_opts=COMPRESSION_OPTS)
        pgrp.create_dataset("indices", data=indices_data,
                            compression=COMPRESSION, compression_opts=COMPRESSION_OPTS)

        print("Rows = " + str(graph.shape[0]))
        print("Cols = " + str(graph.shape[1]))
        print("NonZ = " + str(graph.nnz))

        # return the xmd command and new start address
        return graph.shape[0]



    def prepareGraph(self, graphName, graph, csr, pick):
        # create the HDF5 file for this graph
        os.makedirs(graphDataRoot, exist_ok=True)
        h5_path = os.path.join(graphDataRoot, graphName + ".h5")

        with h5py.File(h5_path, "w") as h5f:
            h5f.attrs["graph_name"] = graphName
            h5f.attrs["format"] = "csr" if csr else "csc"
            h5f.attrs["partition_mode"] = pick

            for num_partition in num_cu:
                print("Graph " + graphName + " with " + str(num_partition) + " partitions")

                # create the graph partitions list
                partitions = []
                if(pick == "row"):
                    partitions = rowSplit(graph,num_partition)
                elif pick == "nnz":
                    partitions = nnzSplit(graph,num_partition)

                # create HDF5 group for this GPU count
                gpu_group = h5f.create_group(f"gpus_{num_partition}")

                # serialize the graph data and build commands
                i = 0

                startRow = 0
                savedRows = 0

                # Build meta arrays
                meta32_list = []
                meta64_list = []

                for i, part in enumerate(partitions):
                    # savedRows = break_idx[i]
                    print ("\n------------\n"+"Partition " + str(i) + "\n------------")
                    # write the metadata base ptr into
                    if (csr):
                        res = self.serializeGraphData(gpu_group, part, graphName + "-" + str(i),
                                                      i,savedRows)
                    else:
                        res = self.serializeGraphData(gpu_group, part, graphName + "-" + str(i),
                                                      i,savedRows)
                    savedRows += res
                    meta32_list.append(np.uint32(part.shape[0]))  # uint32
                    meta32_list.append(np.uint32(part.shape[1]))  # uint32
                    meta64_list.append(np.uint64(part.nnz))       # uint64
                    meta32_list.append(np.uint32(startRow))       # uint32
                    #update start row
                    startRow += part.shape[0]  

                # Write meta arrays into the gpu group
                gpu_group.create_dataset("meta32",
                                         data=np.array(meta32_list, dtype=np.uint32),
                                         compression=COMPRESSION, compression_opts=COMPRESSION_OPTS)
                gpu_group.create_dataset("meta64",
                                         data=np.array(meta64_list, dtype=np.uint64),
                                         compression=COMPRESSION, compression_opts=COMPRESSION_OPTS)
                
                print ("Graph " + graphName + " prepared with " + str(num_partition) + " partitions")
                if csr:
                    print("Matrix stored in row-major format")
                else:
                    print ("Matrix stored in col-major format")

        size_mb = os.path.getsize(h5_path) / (1024 * 1024)
        print ("All data is located in " + h5_path + " (" + f"{size_mb:.2f}" + " MB)\n")


def detect_format(path):
    sep = '\t'
    comment = '#'
    skip = 0
    with open(path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('%'):
                comment = '%'
                skip += 1
            elif stripped.startswith('#'):
                comment = '#'
                skip += 1
            else:
                if comment == '%':
                    skip += 1  # skip the MTX dimension header line
                sep = '\t' if '\t' in stripped else ' '
                break
    return sep, comment, skip

def loadGraph(matrix, dim):
    name_matrix = str(matrix) + '.txt'
    path_to_go = localtxtFolder + name_matrix

    sep, comment, skip = detect_format(path_to_go)
    arr = np.loadtxt(path_to_go, dtype=float, comments=comment,
                     delimiter=sep, usecols=[0, 1], skiprows=skip)

    row = arr[:, 0].astype(np.uint32)
    col = arr[:, 1].astype(np.uint32)
    data = np.ones(len(row), dtype=np.uint32)

    if dim is None:
        dim = int(max(row.max(), col.max())) + 1

    return sparse.csr_matrix((data, (row, col)), shape=(dim, dim))


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
        dim = int(sys.argv[3]) if len(sys.argv) == 4 else None # number of nodes
        buildGraphManagerSingle(dataset_name,dim,partition_mode)