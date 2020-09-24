# has to be run with either sklearn version 0.22 or with updated grakel that imports joblib independently
# On the 24/09/2020 the updated grakel is only available from github (not pip)

import networkx as nx
from grakel.utils import graph_from_networkx
from grakel.kernels import GraphHopper
import numpy as np
from nilearn.image import load_img
import json
from nilearn.datasets import fetch_atlas_yeo_2011
from nilearn.input_data import NiftiLabelsMasker
from scipy.stats import pearsonr


def centeroidnp(arr):
    """
    return the parcel's baricentre

    Parameters
    ----------
    arr : np.array() of shape (?,3)
        three stacked arrays respectively with the x, y, and z coordinates of the voxels in the parcel

    Output
    ------
    x, y, z : coordinates of the baricentre of the parcel
    """
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return sum_x / length, sum_y / length, sum_z / length


def connectivity_fingerprint(yeo_timeseries, wards_timeserie):
    """
    Will calculate connectivity between atlas regions and parcels

    Parameters
    ----------
    yeo_timeseries : np.array, (n_time * 17)
        an rsfMRI seperated in 17 timeseries corresponding to the 17 regions in the yeo (2011) atlas

    wards_timeserie : np.array (n_time)
        a single rsfmri parcel timeserie

    Output
    ------
    corr : (n_parcels * n_atlas_regions) np.array(), should be 200*17
        grid of correlation of activation between a given parcel and a given region in an atlas over a time series
    """
    corr = np.empty(len(yeo_timeseries.T))
    for i in range(len(yeo_timeseries.T)):  # for yeo, we iterate through the 17 parcels
        corr[i] = pearsonr(wards_timeserie, yeo_timeseries[:, i])[0]
    corr[
        np.where(np.isnan(corr[:]))
    ] = 0  # removing nans, of which we get quite a few, when an input array is constant
    return corr


## Create graphs (one graph per subject)
def compute_graph(sub_name, spatial_regulation=10):
    """
    Parameters
    ----------
    sub_name : subject name, should correspond to the folder in which ward data is saved

    spatial_regulation : float,
        regulation used on the spatial addition in the ward_parcellation.
        Used to collect the file, as it appears in it (the XX in skward_regXX_parcellation.nii.gz)

    Output
    ------
    gx : stack of parcel baricentres and yeo-ROI connectivity fingerprint in a graph 
    """
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

    yeo = fetch_atlas_yeo_2011()
    yeo_masker = NiftiLabelsMasker(
        labels_img=yeo["thick_17"], standardize=True, memory="nilearn_cache"
    )
    wards_masker = NiftiLabelsMasker(
        labels_img=wards, standardize=True, memory="nilearn_cache"
    )
    rsfmri_img = load_img(
        "/scratch/mmahaut/data/abide/downloaded_preprocessed/NYU_0051091/NYU_0051091_func_preproc.nii.gz"
    )
    wards_timeseries = wards_masker.fit_transform(rsfmri_img)
    yeo_timeseries = yeo_masker.fit_transform(rsfmri_img)

    for i in range(len(wards_timeseries.T)):
        # compute 3D coordinates of the barycenter of the parcel i
        pos = np.array(np.where(wards_data == i))
        x1_i = centeroidnp(pos)
        X1.append(x1_i)
        # compute connectivity fingerprint of the parcel i
        x2_i = x1_i
        # x2_i = connectivity_fingerprint(yeo_timeseries, wards_timeseries[:, i])
        X2.append(x2_i)
    # optional: normalize X1 and X2 (with min & max)
    X1 = np.array(X1)
    X2 = np.array(X2)

    X1_norm = (X1 - X1.min()) / (X1.max() - X1.min())
    X2_norm = (X2 - X2.min()) / (X2.max() - X2.min())
    # concatenate X1 and X2 to produce the full
    # set of attributes
    attr = np.hstack([X1_norm, X2_norm])
    print(sub_name, " attr : ", attr.shape)
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

    # all your  gx graphs are in a list of graphs called nx_graphs

    # transform networkx-graph into GraKel-graph
    G = list(graph_from_networkx(nx_graphs, node_labels_tag="attributes"))

    gamma = (
        1.0  # I need to check which value we should use... we will change it later...
    )
    print("GraphHopper gamma : {}".format(gamma))
    gk = GraphHopper(normalize=True, kernel_type=("gaussian", float(gamma)))
    K = gk.fit_transform(G)
    np.save("/scratch/mmahaut/data/abide/graph_classification/gram_matrix.npy", K)
    # K is your gram matrix, that you can then use to perform SVM-based classification...
