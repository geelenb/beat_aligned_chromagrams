import itertools
import multiprocessing
import os
import pickle
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

from gtzan import analyse_audio
from ml import dictionary_of_models, accuracy_from_confusion_matrix
from pitches_problem import PitchesProblem
from plotting import imshow_confusion_matrix
from representations import (
    combined_representation,
    represent_w_correlation,
    represent_w_mean,
    represent_w_bpm,
    represent_w_num_beats,
    represent_w_song_length,
)

np.warnings.filterwarnings("ignore")

dataset_root = "/Users/bgeelen/Data/ismir04_genre/"


def pitchesproblem_for_case(dataset_root: str, case: str) -> PitchesProblem:
    csv_fname = os.path.join(dataset_root, "metadata", case, "tracklist.csv")
    db_table = pd.read_csv(
        csv_fname,
        index_col=False,
        names=["label", "artist", "album", "track", "track_number", "file_path"],
    )
    labels_str = db_table.label.values

    i_to_labelname = list(sorted(set(labels_str)))
    labelname_to_i = {labelname: i for i, labelname in enumerate(i_to_labelname)}

    labels_i = [labelname_to_i[label] for label in labels_str]

    # handle one annoying case
    full_fnames = [
        "electronic/strojovna_07/dirnix/05-kruhovy_objazd.mp3"
        if "objazd" in fname
        else fname
        for fname in db_table.file_path.values
    ]

    # prepend the rest of the path
    full_fnames = [
        os.path.join(dataset_root, "audio", case, fname) for fname in full_fnames
    ]

    # handle some cases where there's an ' in the filename
    full_fnames = [
        fname if os.path.exists(fname) else fname.replace("'", "_")
        for fname in full_fnames
    ]

    for i, fname in enumerate(full_fnames):
        if not os.path.exists(fname):
            print("file does not exist:", fname)

    with multiprocessing.Pool() as pool:
        song_analyses = list(
            pool.map(partial(analyse_audio, include_timbres=True), full_fnames)
        )

    return PitchesProblem(song_analyses, np.array(labels_i), i_to_labelname)


if __name__ == "__main__":
    dataset_pickle_fname = "ismir_genre_dataset.p"
    if os.path.exists(dataset_pickle_fname):
        print(f"Reading dataset from pickle file {dataset_pickle_fname}")
        with open(dataset_pickle_fname, "rb") as f:
            casename_to_pitchesproblem = pickle.load(f)
        cases = list(casename_to_pitchesproblem.keys())
    else:
        print(f"Creating dataset by parsing audio files...")

        cases = ["development", "training", "evaluation"]
        casename_to_pitchesproblem = {
            case: pitchesproblem_for_case(dataset_root, case) for case in cases
        }

        with open(dataset_pickle_fname, "wb") as f:
            pickle.dump(casename_to_pitchesproblem, f)

    train_y = casename_to_pitchesproblem["training"].y
    dev_y = casename_to_pitchesproblem["development"].y
    test_y = casename_to_pitchesproblem["evaluation"].y

    labelnames = casename_to_pitchesproblem["training"].labelnames

    repr_funcs = {
        "combined_2_stats": combined_representation(
            represent_w_mean,
            represent_w_bpm,
            represent_w_num_beats,
            represent_w_song_length,
            represent_w_correlation,
            partial(represent_w_correlation, rank=0),
            partial(represent_w_correlation, rank=1),
            partial(represent_w_correlation, rank=2),
            partial(represent_w_correlation, rank=4),
            partial(represent_w_correlation, rank=8),
            partial(represent_w_correlation, rank=16),
        ),
    }

    accuracies = dict()
    name_to_model = dictionary_of_models()
    name_to_model = {
        "XGBoost": XGBClassifier(n_estimators=1000, n_jobs=-1),
    }
    print(
        f"Evaluating {len(repr_funcs)} different representation methods with {len(name_to_model)} models..."
    )

    for repr_func_name, repr_func in repr_funcs.items():
        print(repr_func_name)

        with multiprocessing.Pool() as pool:
            train_X = np.vstack(
                map(repr_func, casename_to_pitchesproblem["training"].song_analyses)
            )
            dev_X = np.vstack(
                map(repr_func, casename_to_pitchesproblem["development"].song_analyses)
            )

        accuracies[repr_func_name] = dict()

        for model_name, model in name_to_model.items():
            print(" ", model_name.ljust(50), end="")

            model.fit(train_X, train_y)
            prediction_y = model.predict(dev_X)

            mat = confusion_matrix(
                dev_y,
                prediction_y,
                # labels=labelnames,
            )

            accuracy = accuracy_from_confusion_matrix(mat)
            accuracies[repr_func_name][model_name] = accuracy
            print(f'{accuracy:.4f}')

    df_accuracies = pd.DataFrame(accuracies)

    best_repr_name, best_model_name = df_accuracies.unstack().argmax()
    print(
        f"Making a model of the best type ('{best_repr_name}', '{best_model_name}')..."
    )
    best_repr_f = repr_funcs[best_repr_name]
    best_model = name_to_model[best_model_name]

    train_X = np.vstack(
        map(
            best_repr_f,
            itertools.chain(
                casename_to_pitchesproblem["training"].song_analyses,
                casename_to_pitchesproblem["development"].song_analyses,
            ),
        )
    )
    train_y = np.concatenate(
        [
            casename_to_pitchesproblem["training"].y,
            casename_to_pitchesproblem["development"].y,
        ]
    )

    test_X = np.vstack(
        map(
            best_repr_f, casename_to_pitchesproblem["evaluation"].song_analyses
        )
    )
    test_y = casename_to_pitchesproblem["evaluation"].y

    best_model.fit(train_X, train_y)
    prediction_y = best_model.predict(test_X)

    mat = confusion_matrix(test_y, prediction_y)

    accuracy = accuracy_from_confusion_matrix(mat)
    print("Evaluation accuracy:", accuracy)

    imshow_confusion_matrix(
        mat,
        [' '.join(labelname.split('_')).title() for labelname in labelnames],
        title="ISMIR-genre: Evaluation set confusion matrix",
        out_fname=None,  # "ismir_genre_mat.svg",
    )
