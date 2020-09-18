import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump

if __name__ == "__main__":
    # the full gram matrix, n_samples x n_samples
    K = np.load("/scratch/mmahaut/data/abide/graph_classification/gram_matrix.npy")

    # the vector of labels
    y = []
    classified_file = open(
        "/scratch/mmahaut/scripts/graph_classification_rsfmri/subs_list_asd_classified.json"
    )  # Hardwriten non-modifiable paths in script is bad practice. modify later !
    classified_dict = json.load(classified_file)
    # no normalisation step (which kind of seems legit for classification)
    for key in classified_dict:
        y.append(1 if classified_dict[key] == "asd" else 0)
        y = np.array(y)

    # define your cross-validation
    cv = KFold(n_splits=10)

    score_list = []
    fold = 0
    for train_index, test_index in cv.split(K):
        fold += 1
        # print("TRAIN:", train_index, "TEST:", test_index)
        K_train = K[np.ix_(train_index, train_index)]
        K_test = K[np.ix_(test_index, train_index)]
        y_train = y[train_index]
        y_test = y[test_index]

        svc = SVC(kernel="precomputed")
        svc.fit(K_train, y_train)
        dump(
            svc,
            "/scratch/mmahaut/data/abide/graph_classification/model_fold{}.sav".format(
                fold
            ),
        )
        y_pred = svc.predict(K_test)
        score = accuracy_score(y_test, y_pred)
        score_list.append(score)

    print("\tscore: {:.02f}".format(np.mean(score_list) * 100))
    np.save(
        "/scratch/mmahaut/data/abide/graph_classification/score_list.npy", score_list
    )
