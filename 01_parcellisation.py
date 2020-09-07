from nilearn.regions import Parcellations
import time


def parcellation(sub_name="NYU_0051091"):
    start = time.time()
    rena = Parcellations(
        method="rena",
        n_parcels=5000,
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


# visualisation and saving : https://nilearn.github.io/auto_examples/03_connectivity/plot_data_driven_parcellations.html#sphx-glr-auto-examples-03-connectivity-plot-data-driven-parcellations-py

