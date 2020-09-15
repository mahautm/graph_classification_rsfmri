import networkx as nx
from grakel.utils import graph_from_networkx
from grakel.kernels import GraphHopper

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return sum_x/length, sum_y/length
## Create graphs (one graph per subject)

# create networkx-graph from the n_parcels x n_parcels adjacency matrix A
gx = nx.from_numpy_matrix(A)

# for each parcel (which will become a graph node), compute two sets of attributes:
X1 = []
X2 = []
for i in range(n_parcels):
    # compute 3D coordinates of the barycenter of the parcel i
    x1_i = centeroidnp(loadparceldata)# TODO loadparceldata
    X1.append(x1_i)
    # compute connectivity fingerprint of the parcel i
    x2_i = x1_i # TODO (but later; to start with, just use x2_i = x1_i)
    X2.append(x2_i)
# optional: normalize X1 and X2 (with min & max)
X1 = # TODO
X2 = # TODO
# concatenate X1 and X2 to produce the full set of attributes
attr = np.hstack([X1_norm, X2_norm]) # TODO: you might need to transpose them!
# construct dict of attributes
d = {i: list(attr[i, :]) for i in range(attr.shape[0])}
# add attributes to nodes
nx.set_node_attributes(gx, d, 'attributes')


## Compute gram matrix

# let's assume that you have put all your graphs gx into a list of graphs called nx_graphs

# transform networkx-graph into GraKel-graph
G = list(graph_from_networkx(nx_graphs, node_labels_tag='attributes'))

gamma = 1. # I need to check which value we should use... we will change it later...
print("GraphHopper gamma : {}".format(gamma))
gk = GraphHopper(normalize=True, kernel_type=('gaussian',float(gamma)))
K = gk.fit_transform(G)

# K is your gram matrix, that you can then use to perform SVM-based classification...
# I will give you another bit of code to do it!
