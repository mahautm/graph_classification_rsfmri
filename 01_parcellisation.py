from nilearn.regions import Parcellations
import time
import os
import json


def parcellation(sub_name="NYU_0051091"):
    start = time.time()
    rena = Parcellations(
        method="rena",
        n_parcels=200,
        standardize=False,
        smoothing_fwhm=2.0,
        scaling=True,
    )

    rena.fit_transform(
        "/scratch/mmahaut/data/abide/downloaded_preprocessed/{}/{}_func_preproc.nii.gz".format(
            sub_name, sub_name
        )
    )
    print("ReNA 5000 clusters: %.2fs" % (time.time() - start))
    rena_labels_img = rena.labels_img_
    # Now, rena_labels_img are Nifti1Image object, it can be saved to file
    # with the following code:
    out_dir = "/scratch/mmahaut/data/abide/graph_classification/{}".format(sub_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    rena_labels_img.to_filename(os.path.join(out_dir, "rena_parcellation.nii.gz"))


if __name__ == "__main__":
    subs_list_file = open("subs_list_asd.json")
    subs_list = json.load(subs_list_file)
    for sub_name in subs_list:
        parcellation(sub_name)
