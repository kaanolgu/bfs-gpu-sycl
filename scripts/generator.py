#!/usr/bin/env python


import io, sys, numpy, scipy, struct, os
from scipy import io
from scipy import sparse
from numpy import inf
import numpy as np
import networkx as nx
# initialize access to UF SpM collection
# import yaUFget as uf
def round8(a):
    return int(a) + 4 & ~7

dramBase = 0x10000000

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
    # print(nnz)
    # print("shape[0]: ",matrix.shape[0])
    # for i in range(0, matrix.shape[0]):
    # print(matrix.getrow(1013).toarray()[0])
    total = nnz[-1]
    # print(total)
    ideal_breaks = np.arange(0, total, total/n_compute_units)
    # print(ideal_breaks)
    break_idx = [*nnz.searchsorted(ideal_breaks),matrix.shape[0]]
    # make sure that break_idx is divisible by NUM_BITS_VISITED
    break_idx = [round8(x) for x in break_idx]
    # print(break_idx)
    return [
        matrix[i: j,:]
        for i, j in zip(break_idx[:-1], break_idx[1:])
    ]

def rowSplit(
    matrix: sparse.sparray, n_compute_units: int = 4,
) -> list[sparse.sparray]:
    # print(nnz)
    # print("shape[0]: ",matrix.shape[0])
    # for i in range(0, matrix.shape[0]):
    # print(matrix.getrow(1013).toarray()[0])
    total = matrix.shape[0]
    # print("total: ",total)
    stepSize = int(alignedIncrement(total / n_compute_units,0,64))
    print(stepSize)
    break_idx = np.arange(0, total, stepSize)
    break_idx = np.append(break_idx,total)
    # print("ideal_breaks: ",break_idx)
 
    print(matrix.shape[0])
    print(break_idx)
    # make sure that break_idx is divisible by NUM_BITS_VISITED
    break_idx = [round8(x) for x in break_idx]
    print(break_idx)
    return [
        matrix[i: j,:]
        for i, j in zip(break_idx[:-1], break_idx[1:])
    ]




    # submatrices = []
    # # align partition size (rowcount) to 64 - good for bursts and
    # # updating the input vector bits independently
    # stepSize = int(alignedIncrement(csr.shape[0] / numPartitions,0,64))
    # print(csr.shape[0])
    # print(numPartitions)
    # print(stepSize)
    # print("=======================\n")
    # # create submatrices, last one includes the remainder
    # currentRow = int(0)
    # for i in range(0, numPartitions):
    #     if i != numPartitions - 1:
    #         submatrices += [csr[currentRow:currentRow + stepSize]]  # essentially this is initally csr[0:0+stepsize] = csr[0:stepsize]  
    #     else:
    #         submatrices += [csr[currentRow:]]
    #     currentRow += stepSize
    # return submatrices

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

def buildGraphManager(c,pick):
    graphs = makeGraphList()
    for g in graphs:
        m = GraphMatrix()
        print ("Graph " + g + " with " + str(c) + " partitions")
        m.prepareGraph(g, c,False,pick)
      
def buildGraphManagerSingle(name, pick):
    g = name
    m = GraphMatrix()


    # Load the graph once outside the loop
    graph = loadGraph(g)
    print("PATH : ", localtxtFolder + g + ".txt")

    # Call prepareGraph with the loaded graph for each partition count
    for num_partition in num_cu:
        print("Graph " + g + " with " + str(num_partition) + " partitions")
        m.prepareGraph(g, num_partition, graph, False, pick)
      
   

'''
def removeSelfEdges(graph):
  lil=scipy.sparse.lil_matrix(graph)
  lil.setdiag([0 for i in range(0,graph.shape[0])])
  if scipy.sparse.isspmatrix_csr(graph): 
    return scipy.sparse.csr_matrix(lil)
  elif scipy.sparse.isspmatrix_csc(graph):
    return scipy.sparse.csc_matrix(lil)
'''


# as the G500 spec says we should also count self-loops,
# removeSelfEdges does not do anything.
def removeSelfEdges(graph):
    return graph


class GraphMatrix:
    def __init__(self):
        self.copyCommandBuffer = []
        self.graphName = ""

    def resetCommandBuffer(self):
        self.copyCommandBuffer = []



    def serializeGraphData(self, graph, name, rootFolder, startAddr, startRow,index,PrevRowsValue):

        # A = loadGraph(graph)
        A = graph

        # # save metadata
        # fileName = rootFolder + "/" + name + "-meta.bin"
        # metaDataFile = open(fileName, "wb")

      


        # save index pointers
        fileName = rootFolder + "/" + name + "-indptr.bin"
        indPtrFile = open(fileName, "wb")
        indPtrFile.write(A.indptr)
        indPtrFile.close()

        # save indptr data start into metadata
        


        # save indices
        fileName = rootFolder + "/" + name + "-inds.bin"
        indsFile = open(fileName, "wb")
        A.indices = A.indices + PrevRowsValue
        # A.indices = A.indices 
        indsFile.write(A.indices)
        indsFile.close()
        # print(A.indices[A.indptr[1013]])
        # save inds data start into metadata
        # difference_ptrs = A.indptr[1014]-A.indptr[1013]
        # print("node[1013] neighbours in this partition : ", difference_ptrs)
        print("---")
        # for i in range(difference_ptrs):
        #     print(A.indices[A.indptr[1013] + i])
        # print("---")
        print("Rows = " + str(A.shape[0]))
        print("Cols = " + str(A.shape[1]))
        print("NonZ = " + str(A.nnz))
        # metaDataFile.write(struct.pack("I", A.shape[0]))
        # metaDataFile.write(struct.pack("I", A.shape[1]))
        # metaDataFile.write(struct.pack("I", A.nnz))
        # metaDataFile.write(struct.pack("I", startRow))  
        # metaDataFile.close()
       
        # return the xmd command and new start address
        return [A.shape[0], A.nnz]



    def prepareGraph(self, graphName, partitionCount, graph, csr, pick):
        if (csr):
            graphName += "-" + pick + "-csr"
        else:
            graphName += "-" + pick + "-csc"
            # SpMV BFS needs transpose of matrix
            graph = graph.transpose()

        graphName = graphName.replace("/", "-")

        # create the graph partitions
        # 
        # creating list
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

#struct.pack converts integer("I") to binary
        startRow = 0
        startAddr = 0
        savedRows = 0
        metafileName = targetDir + "/" + graphName + "-meta.bin"
        metaDataFile = open(metafileName, "wb")

        for i in range(0, partitionCount):
            # savedRows = break_idx[i]
            print ("Partition " + str(i))
            # write the metadata base ptr into
            if (csr):
                res = self.serializeGraphData(partitions[i], graphName + "-" + str(i),
                                              targetDir, startAddr, startRow,i,savedRows)
            else:
                res = self.serializeGraphData(partitions[i], graphName + "-" + str(i),
                                              targetDir, startAddr, startRow,i,savedRows)
            
            # print(startRow)
            startAddr = res[1]
            savedRows += res[0]
            metaDataFile.write(struct.pack("I", partitions[i].shape[0]))
            metaDataFile.write(struct.pack("I", partitions[i].shape[1]))
            metaDataFile.write(struct.pack("I", partitions[i].nnz))
            metaDataFile.write(struct.pack("I", startRow))
            #update start row
            startRow += partitions[i].shape[0]  
            
            # print("saved rows : ", savedRows)
      
            # i += 1
        
        metaDataFile.close()
        print ("Graph " + graphName + " prepared with " + str(partitionCount) + " partitions")
        if csr:
            print("Matrix stored in row-major format")
        else:
            print ("Matrix stored in col-major format")
        print ("All data is located in " + targetDir + "\n")

        self.graphName = graphName
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

def loadGraph(matrix):
    name_matrix = str(matrix) + '.txt'
    path_to_go = localtxtFolder + name_matrix
    
    # print(path_to_go)
    if scipy.sparse.isspmatrix_csc(matrix) or scipy.sparse.isspmatrix_csr(matrix):
        # return already loaded matrix
        r = removeSelfEdges(matrix)
        # do not adjust dimensions, return directly
        return r
    else:
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
            arr = np.loadtxt(path_to_go, dtype=int, comments='%', skiprows=num_comment_lines + 1)
        else:
            arr = np.loadtxt(path_to_go, dtype=int,comments=['#', '$'])
        # print(r)
        # print(arr)
    # else:
    # load matrix from University of Florida sparse matrix collection
    # r=removeSelfEdges(uf.get(matrix)['A'])
    # graph must have rows==cols, clip matrix if needed
   
    test = np.ones((len(arr[:, 0]),), dtype=int)
    arr = sparse.csr_matrix(((test,((arr[:, 0],(arr[:, 1]))))))
    #print r
    rows = arr.shape[0]
    cols = arr.shape[1]
    # print(rows)
    #print cols
    if rows != cols:
        dim = min(rows, cols)
        arr = arr[0:dim, 0:dim]
    return arr


# Slice matrix along rows to create desired number of partitions
# Keeps number of rows equal, does not look at evenness of NZ distribution
def horizontalSplit(matrix, numPartitions):
    csr = matrix.tocsr()
    # print(csr)
    print ('Test=================================================================')
    submatrices = []
    # align partition size (rowcount) to 64 - good for bursts and
    # updating the input vector bits independently
    stepSize = int(alignedIncrement(csr.shape[0] / numPartitions,0,64))
    print(csr.shape[0])
    print(numPartitions)
    print(stepSize)
    print("=======================\n")
    # create submatrices, last one includes the remainder
    currentRow = int(0)
    for i in range(0, numPartitions):
        if i != numPartitions - 1:
            submatrices += [csr[currentRow:currentRow + stepSize]]  # essentially this is initally csr[0:0+stepsize] = csr[0:stepsize]  
        else:
            submatrices += [csr[currentRow:]]
        currentRow += stepSize
    return submatrices




# increment base address by <increment> and ensure alignment to <align>
def alignedIncrement(base, increment, align):
    res = base + increment
    rem = res % align
    if rem != 0:
        res += align - rem
    return res

if __name__ == '__main__':
    if(sys.argv[1] == "all"):
        for num_partition in num_cu:
            buildGraphManager(num_partition,sys.argv[2])
        # if(sys.argv[2] == "genrootnodes"):
        #     emp_num = dict()
        #     emp_num = buildRootNodes(16)
        #     print(emp_num)
    else:
        dataset_name = sys.argv[1]
        partition_mode = sys.argv[2] ## nnz or row
        buildGraphManagerSingle(dataset_name,partition_mode)
        # print("======== Generate Root Nodes =======")
        # emp_num = dict()
        # emp_num = buildRootNodesSingle(sys.argv[1],2)
        # print(emp_num)