import networkx as nx
from grakel.utils import graph_from_networkx
from grakel.kernels import GraphHopper
import numpy as np
from nilearn.image import load_img
import json


def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return sum_x / length, sum_y / length, sum_z / length


## Create graphs (one graph per subject)
def compute_graph(sub_name, spatial_regulation=10):
    wards = load_img(
        "/scratch/mmahaut/data/abide/graph_classification/{}/skward_reg{}_parcellation.nii.gz".format(
            sub_name, spatial_regulation
        )
    )
    wards_data = wards.get_fdata()
    A = np.load(
        "/scratch/mmahaut/data/abide/graph_classification/{}/parcel_adjacency.npy".format(
            sub_name
        )
    )
    # create networkx-graph from the n_parcels x n_parcels adjacency matrix A
    gx = nx.from_numpy_matrix(A)
    n_parcels = A.shape[0]
    # for each parcel (which will become a graph node), compute two sets of attributes:
    X1 = []
    X2 = []
    for i in range(n_parcels):
        # compute 3D coordinates of the barycenter of the parcel i
        pos = np.where(wards_data == i)
        x1_i = centeroidnp(pos)
        X1.append(x1_i)
        # compute connectivity fingerprint of the parcel i
        x2_i = x1_i  # TODO (but later; to start with, just use x2_i = x1_i)
        X2.append(x2_i)
    # optional: normalize X1 and X2 (with min & max)
    X1 = np.array(X1)
    X2 = np.array(X2)

    X1_norm = (X1 - X1.min()) / (X1.max() - X1.min())
    X2_norm = (X2 - X2.min()) / (X2.max() - X2.min())
    # concatenate X1 and X2 to produce the full set of attributes
    attr = np.hstack([X1_norm, X2_norm])  # TODO: you might need to transpose them!
    # construct dict of attributes
    d = {i: list(attr[i, :]) for i in range(attr.shape[0])}
    # add attributes to nodes
    nx.set_node_attributes(gx, d, "attributes")
    return gx


if __name__ == "__main__":
    subs_list_file = open("subs_list_asd.json")
    subs_list = json.load(subs_list_file)
    nx_graphs = list()
    for sub_name in subs_list:
        nx_graphs.append(compute_graph(sub_name))
## Compute gram matrix

# let's assume that you have put all your graphs gx into a list of graphs called nx_graphs

# transform networkx-graph into GraKel-graph
G = list(graph_from_networkx(nx_graphs, node_labels_tag="attributes"))

gamma = 1.0  # I need to check which value we should use... we will change it later...
print("GraphHopper gamma : {}".format(gamma))
gk = GraphHopper(normalize=True, kernel_type=("gaussian", float(gamma)))
K = gk.fit_transform(G)

# K is your gram matrix, that you can then use to perform SVM-based classification...
# I will give you another bit of code to do it!
