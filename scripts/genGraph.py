# import snap
import sys,os
import numpy as np
import networkit as nk
from array import array
from scipy.io import savemat


graph_identifier = (sys.argv[1])

absolute_path = os.path.dirname(__file__)
relative_path_localtxt = "txt/"
relative_path_localmat = "matlab/"
localtxtFolder = os.path.join(absolute_path, relative_path_localtxt)
localmatFolder = os.path.join(absolute_path, relative_path_localmat)

def makeGraphList():
    graphs = []
    # Get all the files ending with .mat extension 
    for file in os.listdir(localtxtFolder):
    	if file.endswith(".txt"):
        	graphs += [file.rsplit( ".", 1 )[0]]
    print(graphs)
    # graphs = ["rmat-19-32"]
    return graphs

def makeCustomGraphList():
    graphs = []
    # Get all the files ending with .mat extension 
    for file in os.listdir(localtxtFolder):
    	if file.endswith(".txt"):
                if not (file.startswith("rmat")):
        	        graphs += [file.rsplit( ".", 1 )[0]]
    print(graphs)
    # graphs = ["rmat-19-32"]
    return graphs




if(graph_identifier == "rmat"):
    lines=[]
    # Rnd = snap.TRnd()
    S = int(sys.argv[2]) # scale factor
    E = int(sys.argv[3]) # edge factor


    # Benchmark Parameters (Do not modify these are from Graph500 website)
    A = 0.57
    B = 0.19
    C = 0.19
    D = 1-(A+B+C) 

    # Generate Graph
    rmat = nk.generators.RmatGenerator(S, E, A, B, C, D,False,0)
    rmatG = rmat.generate()
    
    print(rmatG.numberOfNodes(), rmatG.numberOfEdges(), rmatG.isDirected())

    myfile = open(localtxtFolder+'rmat-'+str(S)+'-'+str(E)+'.txt', "w")
    for u, v, w in rmatG.iterEdgesWeights():
        lines=str(u)+"\t"+str(v)+"\t"+str(int(w))+"\n"
        myfile.write(str(lines))
        # print("edge: (%d, %d)" % (u, v))


    # arr = np.loadtxt(localtxtFolder+'RMAT-'+str(S)+'-'+str(E)+'.txt', dtype=int)
    # mdic = {"M": arr}
    # savemat(localmatFolder+'rmat-'+str(S)+'-'+str(E) +".mat", mdic)
    print(localtxtFolder+"rmat-"+str(S)+"-"+str(E) +".txt : SUCCESS!")
else:
    g=graph_identifier
    with open(localtxtFolder+g+".txt") as ifh, open(localtxtFolder+g.upper() + ".txt", 'w+') as ofh:
        for lineno, line in enumerate(ifh):
            if not (line.startswith("#")):
                line = line.rstrip()                 # remove newline
                color = str(1)                         # choose color
                line += '\t' + color                  # append color
                ofh.write(line + '\n')               # write line
    print("3 Column SUccesfull!")
    # arr = np.loadtxt(localtxtFolder+g.upper() + ".txt", dtype=int)
    # mdic = {"M": arr}
    # savemat(localmatFolder + g + ".mat", mdic)
    # print(localmatFolder+ g + ".mat: SUCCESS!")
    # print("Generate (CUSTOM)Mat File SUccesfull!")
    
