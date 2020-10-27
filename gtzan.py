import glob
import multiprocessing
import os
import pickle
from functools import partial

import numpy as np
import pandas as pd

from ml import (
    dictionary_of_models,
    confusion_matrix_for_problem,
    accuracy_from_confusion_matrix,
    cross_val_predict_confusion_matrix_for_problem,
)
from pitches_problem import PitchesProblem
from pitches_problem import analyse_audio
from plotting import imshow_confusion_matrix
from representations import (
    represent_w_mean,
    combined_representation,
    represent_w_correlation,
    represent_w_bpm,
    represent_w_song_length, represent_w_num_beats, represent_w_dct, represent_w_lstsq, represent_w_lstsq_order)

np.warnings.filterwarnings("ignore")


def make_problem(dataset_root, include_timbres):
    fnames = glob.glob(os.path.join(dataset_root, "**", f"*.wav"))
    label_str = [fname.split(os.path.sep)[-2] for fname in fnames]

    i_to_classname = list(sorted(set(label_str)))
    classname_to_i = {cn: i for i, cn in enumerate(i_to_classname)}

    y = np.array([classname_to_i[cn] for cn in label_str])

    f = partial(analyse_audio, include_timbres=include_timbres)
    if True:
        with multiprocessing.Pool() as pool:
            list_of_pitch_matrices = list(pool.map(f, fnames))
    else:
        list_of_pitch_matrices = [f(fname) for fname in fnames]

    return PitchesProblem(list_of_pitch_matrices, y, i_to_classname)


if __name__ == "__main__":
    dataset_root = "/Users/bgeelen/Data/GTZAN/genres"

    database_p = "gtzan.p"
    if os.path.exists(database_p):
        print(f"Reading database from {database_p}")
        with open(database_p, "rb") as f:
            problem = pickle.load(f)
    else:
        print("Creating database...")
        problem = make_problem(dataset_root, True)

        print(f"writing database to {database_p}")
        with open(database_p, "wb") as f:
            pickle.dump(problem, f)

    song_analyses, y, labelnames = problem.song_analyses, problem.y, problem.labelnames

    repr_funcs = {
        # "correlation": represent_w_correlation,
        # "combined_1": combined_representation(
        #     represent_w_correlation, represent_w_mean
        # ),
        # "combined_1_w_bpm": combined_representation(
        #     represent_w_bpm, represent_w_correlation, represent_w_mean
        # ),
        # "combined_2": combined_representation(
        #     represent_w_mean,
        #     represent_w_correlation,
        #     partial(represent_w_correlation, order=2),
        #     partial(represent_w_correlation, order=4),
        # ),
        "combined_with_correlation": combined_representation(
            represent_w_bpm,
            represent_w_song_length,
            represent_w_num_beats,
            represent_w_mean,
            partial(represent_w_correlation, order=0),
            represent_w_correlation,
            partial(represent_w_correlation, order=2),
            partial(represent_w_correlation, order=4),
        ),
    }

    accuracies = dict()
    name_to_model = dictionary_of_models()

    print(
        f"Evaluating {len(repr_funcs)} different representation methods on {len(name_to_model)} models..."
    )

    for repr_func_name, repr_func in repr_funcs.items():
        print(repr_func_name)

        with multiprocessing.Pool() as pool:
            X = np.vstack(map(repr_func, song_analyses))

        accuracies[repr_func_name] = dict()

        for model_name, model in name_to_model.items():
            print(" ", model_name.ljust(50), end="")
            mat = confusion_matrix_for_problem(X, y, model)
            accuracy = accuracy_from_confusion_matrix(mat)
            accuracies[repr_func_name][model_name] = accuracy
            print(f"{accuracy:.4}")

    df_accuracies = pd.DataFrame(accuracies)

    best_repr_name, best_model_name = df_accuracies.unstack().argmax()
    print(
        f"Making a model of the best type ('{best_repr_name}', '{best_model_name}')..."
    )
    best_repr_f = repr_funcs[best_repr_name]
    best_model = name_to_model[best_model_name]

    X = np.vstack(map(best_repr_f, song_analyses))

    mat = cross_val_predict_confusion_matrix_for_problem(X, y, best_model)
    accuracy = accuracy_from_confusion_matrix(mat)
    print("cross validation accuracy:", accuracy)

    imshow_confusion_matrix(
        mat,
        [classname.title() for classname in labelnames],
        title="GTZAN: Cross validation confusion matrix",
        out_fname="gtzan_cv_mat.svg",
    )


#%%




