from nilearn.image import load_img, new_img_like
from nilearn import plotting
from nilearn.input_data import NiftiMasker
from sklearn.feature_extraction.image import grid_to_graph
import numpy as np
import json
from sklearn.cluster import AgglomerativeClustering
import os
import sys
import matplotlib.pyplot as plt
from nilearn.datasets import load_mni152_brain_mask
from scipy.sparse import save_npz


def parcellation(
    sub_name, input_dir, out_dir, spatial_regularisation=10, n_clusters=200
):
    """
    Will load a functional image, and parcellate the brain into different zones that tend to activate together

    Parameters
    ----------
    sub_name : string
        name of the subject, used to access and save his data.

    input_dir : string
        path to where functional images can be loaded from

    out_dir : string
        path where files will be saved once computed

    spatial_regularisation : float, default 10
        once normalised, the spatial parameters a

    n_clusters : int, default 200
        The number of clusters to divide the image into

    Notes
    ------
    the "SUBJECTNAME_func_minimal.nii.gz" rsfMRI file is expected to be found in the input_dir,
    in a folder named after the subject 

    no output, but saves three files in the out_dir, in the corresponding subject file 
    skward_regXX_parcellation.nii.gz : the ward parcellated file
    ward_connectivity.npz : a sparse spatial adjacency matrix, voxel to voxel
    ward_labels.npy


    """
    raw_img = load_img(
        os.path.join(input_dir, "{0}/{0}_func_minimal.nii.gz".format(sub_name)), 1
    )
    flat_img = (raw_img.get_fdata()).reshape(-1, raw_img.shape[3])

    mask_img = load_mni152_brain_mask()
    nifti_masker = NiftiMasker(
        mask_img=mask_img, memory="nilearn_cache", memory_level=1
    )
    nifti_masker = nifti_masker.fit()
    flat_img = nifti_masker.transform(raw_img)
    plotting.plot_roi(mask_img, title="mask", display_mode="xz", cmap="prism")
    mask = nifti_masker.mask_img_.get_fdata().astype(np.bool)
    shape = mask.shape
    connectivity = grid_to_graph(n_x=shape[0], n_y=shape[1], n_z=shape[2], mask=mask)

    # weigh by coords
    x_ = np.arange(61)
    y_ = np.arange(73)
    z_ = np.arange(61)

    coords = np.array(np.meshgrid(x_, y_, z_, indexing="ij"))

    assert np.all(coords[0, :, 0, 0] == x_)
    assert np.all(coords[1, 0, :, 0] == y_)
    assert np.all(coords[2, 0, 0, :] == z_)

    coods_ni = new_img_like(raw_img, np.array(coords).T)
    flat_coords = nifti_masker.transform(coods_ni)
    # normalize
    flat_img = (flat_img - flat_img.min()) / (flat_img.max() - flat_img.min())
    flat_coords = (flat_coords - flat_coords.min()) / (
        flat_coords.max() - flat_coords.min()
    )
    # add coords to data
    input_data = np.vstack((flat_img, spatial_regularisation * flat_coords))

    # Compute clustering
    ward = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="ward", connectivity=connectivity
    )
    ward.fit(input_data.T)
    labels = nifti_masker.inverse_transform(ward.labels_)

    labels.to_filename(
        os.path.join(
            out_dir,
            sub_name,
            "skward_reg{}_parcellation.nii.gz".format(spatial_regularisation),
        )
    )

    save_npz(
        os.path.join(out_dir, sub_name, "ward_connectivity.npz",),
        ward.connectivity.tocsr(),
    )

    np.save(os.path.join(out_dir, sub_name, "ward_labels.npy",), ward.labels_)


if __name__ == "__main__":
    sub_name = sys.argv[1]
    parcellation(
        sub_name,
        "/scratch/mmahaut/data/abide/downloaded_preprocessed/",
        "/scratch/mmahaut/data/abide/graph_classification/",
    )
