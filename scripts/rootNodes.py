#!/usr/bin/env python


import io, sys, numpy, scipy, struct, os
from scipy import io
from scipy import sparse
from numpy import inf
import numpy as np
import networkx as nx

absolute_path = os.path.dirname(__file__) + "/../../wip-bfs-fpga/scripts"
relative_path = "dataset/"
relative_path_localroot = "txt/"
localRoot = os.path.join(absolute_path, relative_path_localroot)
relative_path_localtxt = "txt/"
localtxtFolder = os.path.join(absolute_path, relative_path_localtxt)




def makeGraphList():
    graphs = []
    # Get all the files ending with .mat extension 
    for file in os.listdir(localRoot):
    	if file.endswith(".txt"):
        	graphs += [file.rsplit( ".", 1 )[0]]
    # print(graphs)
    # graphs = ["rmat-19-32"]
    return graphs





def buildGraphManager(c):
    graphs = makeGraphList()
    for g in graphs:
        m = GraphMatrix()
        print ("Graph " + g + " with " + str(c) + " partitions")
        m.prepareGraph(g, c)
      
def buildGraphManagerSingle(name,c):
    g=name
    m = GraphMatrix()
    print ("Graph " + g + " with " + str(c) + " partitions")
    m.prepareGraph(g, c,False)
      

# as the G500 spec says we should also count self-loops,
# removeSelfEdges does not do anything.
def removeSelfEdges(graph):
    return graph
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
        file_names_from_mtx = [
            "webbase-1M.txt",
            "kron_g500-logn21.txt",
            "indochina-2004.txt",
            "hollywood-2009.txt",
            "europe_osm.txt"
        ]
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


#################### Generate ROOT Nodes #################
# def buildRootNodes():
#     print("HERE")
#     graphs = makeGraphList()
#     rnl = dict()
#     for g in graphs:
#         print("Generating root nodes for " + str(g) + "\n===========================\n")
#         rnl[g] = generateRootNodes(g)
#     return rnl

def buildRootNodesSingle(name):
    rnl = dict()
    g=name
    print("Generating root nodes for " + str(g) + "\n===========================\n")
    rnl[g] = generateRootNodes(g)
    return rnl

def find_top_5_max_degree_nodes(G):
    """Returns the top 10 nodes with the highest degrees and their degrees."""
    # Get the top 10 nodes by degree
    top_5_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]
    return top_5_nodes  # Returns a list of tuples (node, degree)

def find_neighbors_in_partitions(G, node):
    """Find the neighbors of a node (without partitioning)."""
    neighbors = set(G.neighbors(node))  # Use a set for faster lookups
    return neighbors



def generateRootNodes(graph):
    A = loadGraph(graph)  # Assuming this function loads the graph correctly
    rootNodes = []
    G = nx.from_scipy_sparse_array(A)
    # Step 1: Find the top 10 nodes with the greatest number of neighbors and their degrees
    top_5_nodes_with_degrees = find_top_5_max_degree_nodes(G)

    # Step 2: Iterate over the top 10 nodes to find their diameter and neighbors' distribution
    for node, degree in top_5_nodes_with_degrees:
        neighbors = find_neighbors_in_partitions(G, node)  # Get neighbors without partitioning
        print(f"Node {node} and has {degree} neighbors.")
        rootNodes.append(node)

    return rootNodes


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # if(sys.argv[1] == "all"):
        # num_partition = int(sys.argv[2])
        # buildGraphManager(num_partition)
        # if(sys.argv[2] == "genrootnodes"):
        #     print("TEST")
        #     emp_num = dict()
        #     emp_num = buildRootNodes()
        #     print(emp_num)
    # else:
        # num_partition = int(sys.argv[2])
        # buildGraphManagerSingle(sys.argv[1],num_partition)
        print("======== Generate Root Nodes =======")
        emp_num = dict()
        emp_num = buildRootNodesSingle(sys.argv[1])
        print(emp_num)