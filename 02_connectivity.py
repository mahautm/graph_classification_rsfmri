from scipy.sparse import coo_matrix
import os
import numpy as np
import json
import sys

# get coord adjacency matrix
def connectivity(sub_name, out_dir, n_clusters=200):
    parcel_submask = []
    parcels_size = []
    for i in range(n_clusters):
        ward_labels = np.load(
            os.path.join(out_dir, sub_name, "ward_labels.npy"), allow_pickle=True
        )
        parcel_submask.append(np.array(np.nonzero(ward_labels == i))[0])
        parcels_size.append(len(parcel_submask[-1]))
    A = np.zeros([n_clusters, n_clusters])
    ward_connectivity = np.load(
        os.path.join(out_dir, sub_name, "ward_connectivity.npy"), allow_pickle=True
    )
    for i in range(n_clusters):
        for j in range(i):
            a = ward_connectivity[parcel_submask[i], :][:, parcel_submask[j]]
            a = ward_connectivity[parcel_submask[i], :][:, parcel_submask[j]]

            A[i, j] = a.sum()
            A[j, i] = A[i, j]
        A[A != 0] = 1
    np.save(os.path.join(out_dir, "parcel_adjacency.npy"), A)


if __name__ == "__main__":
    sub_name = sys.argv[1]
    connectivity(sub_name, "/scratch/mmahaut/data/abide/graph_classification/")

