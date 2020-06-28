import multiprocessing
from typing import Tuple, List

import tqdm
from xgboost import XGBClassifier

from pitches_problem import SongAnalysis, PitchesProblem

# Folder that contains jrp-scores;
# git clone https://github.com/josquin-research-project/jrp-scores --recursive
# make update
git_folder_root = "/Users/bgeelen/data/josquin/source_new/jrp-scores"

# wget https://josquin.stanford.edu/cgi-bin/jrp?a=worklist-json -O worklist.json
# json_filename = "/Users/bgeelen/data/josquin/source_new/worklist.json"

import glob
import os

import music21
import numpy as np
import pandas as pd
import music21.converter
from sklearn.linear_model import RidgeClassifier

from ml import dictionary_of_models, confusion_matrix_for_problem, accuracy_from_confusion_matrix
from representations import represent_w_correlation, represent_w_lstsq, represent_w_lstsq_rank
#%%

def agn_to_genre(agn: str) -> str:
    agn = agn.replace("Chanson", "Song")
    genres = ["Song", "Mass", "Motet"]
    indices = dict()
    for genre in genres:
        if agn.startswith(genre):
            return genre
        if genre in agn:
            indices[genre] = agn.index(genre)

    return pd.Series(indices).argmin()


cache_file = "parse_cache.p"


def attribution_levels_for_work(work):  # -> dict[str, Tuple[int, str]]
    attributions = [
        x.comment[len("attribution-level@") :]
        for x in work.recurse()
        if isinstance(x, music21.humdrum.spineParser.GlobalComment)
        and x.comment.startswith("attribution-level@")
    ]

    return {
        attribution[:3]: (int(attribution[5]), attribution[7:])
        for attribution in attributions
    }


def analyse_work(fname) -> SongAnalysis:
    work = music21.converter.parse(fname)
    id = os.path.split(fname)[1].split("-")[0]

    notes = [x for x in work.recurse() if isinstance(x, music21.note.Note)]
    first_measure = work.measures(0, 1)[0]
    measure_length = first_measure.quarterLength

    if (measure_length % 3.0) % 1.0 == 0.0:
        tactus_length = measure_length / 3
    else:
        tactus_length = measure_length / 2

    X = np.zeros((int(work.quarterLength / tactus_length), 12))
    for note in notes:
        try:
            offset = note.getOffsetInHierarchy(work)
        except KeyError:
            print(fname)
            raise
        X[
            int(offset // tactus_length) : int(
                (offset + note.quarterLength) / tactus_length
            ),
            note.pitch.midi % 12,
        ] += min(1.0, note.quarterLength / tactus_length)
        # todo: check of midden van een noot eindigt in het midden van een tactus

    globalreferences = {
        ref.code: ref.value
        for ref in work.recurse()
        if isinstance(ref, music21.humdrum.spineParser.GlobalReference)
    }

    agn = globalreferences["AGN"]
    genre = agn_to_genre(agn)
    composer = id[:3].lower()

    return SongAnalysis(
        beats=X,
        duration=work.duration.quarterLength,
        source_fname=fname,
        details={
            "id": id,
            "repid": id[:3],
            "agn": agn,
            **globalreferences,
            "attributions": attribution_levels_for_work(work),
        },
        genre=genre,
        composer=composer,
    )


all_genres = {"mass", "motet", "song"}
all_composers = {
    "Agr",
    "Ano",
    "Bru",
    "Bus",
    "Com",
    "Das",
    "Duf",
    "Fry",
    "Fva",
    "Gas",
    "Isa",
    "Jap",
    "Jos",
    "Mar",
    "Mou",
    "Obr",
    "Ock",
    "Ort",
    "Pip",
    "Reg",
    "Rue",
    "Tin",
}
all_composers = {c.lower() for c in all_composers}


def make_problem(
    git_folder_root: str = git_folder_root,
    select_genres: list = None,
    select_composers: list = None,
    objective: str = "genre",
    josquin_unsure_is_ano=False,
) -> PitchesProblem:
    """
    Args:
        git_folder_root:
        json_filename:
        select_genres: list of (the 3) genres to retain, e.g. ['Mass', 'Motet', 'Song']. Use None to select all.
        select_composers: list of composer repids to retain, e.g. ['Jos', 'Rue', 'Ano]. Use None to select all.
        objective: 'genre' or 'composer'

    Returns:
        PitchesProblem
    """
    if objective.lower() not in {"genre", "composer"}:
        raise ValueError(
            f'chromae_dataset "objective" parameter should be one of "genre", "composer". Instead it was "{objective}"'
        )

    if select_genres is not None:
        select_genres = {c.lower() for c in select_genres}
        if len(all_genres.intersection(select_genres)) == 0:
            raise ValueError(
                f"The select_genres ({select_genres}) are not a selection of {all_genres}"
            )

    if select_composers is not None:
        select_composers = {c.lower() for c in select_composers}
        if len(all_composers.intersection(select_composers)) == 0:
            raise ValueError(
                f"The select_composers ({select_composers}) are not a selection of {all_composers}"
            )

    fnames = glob.glob(os.path.join(git_folder_root, "**", "*.krn"), recursive=True)
    ids = [os.path.split(fname)[1].split("_")[0] for fname in fnames]

    # filter the composers
    if select_composers is not None:
        selected_fnames = {fname for id, fname in zip(ids, fnames) if id[:3].lower() in select_composers}

        if josquin_unsure_is_ano and "Ano" in select_composers:
            selected_fnames.update({fname for id, fname in zip(ids, fnames) if id[:3].lower() == "jos"})

        fnames = selected_fnames

    analyses = [analyse_work(fname) for fname in tqdm.tqdm(fnames)]

    # filter the genres
    if select_genres is not None:
        analyses = [sa for sa in analyses if sa.genre in select_genres]


    if josquin_unsure_is_ano:
        for sa in analyses:
            # 1 is "secure", 2 is "slightly less secure"
            if sa.details["attributions"].get("Jos", [0])[0] > 2:
                sa.composer = "ano"

        if select_composers is not None:
            analyses = [sa for sa in analyses if sa.composer in select_composers]
    if objective.lower() == "genre":
        labelnames, y = list_to_labelnames_and_y([a.genre for a in analyses])
    else:
        labelnames, y = list_to_labelnames_and_y([a.composer for a in analyses])

    return PitchesProblem(analyses, y, labelnames)


def list_to_labelnames_and_y(l: List[str]) -> Tuple[List[str], np.ndarray]:
    labelnames = list(sorted(set(l)))
    label_to_i = {label: i for i, label in enumerate(labelnames)}
    y = np.array([label_to_i[x] for x in l])
    return labelnames, y


# %%
if __name__ == "__main__":
    problem = make_problem(
        objective="composer",
        josquin_unsure_is_ano=True,
    )
        # select_composers=['Jos', 'Rue', 'Mar', 'Ock', 'Ano'],

    repr_funcs = {"correlation": lambda x: represent_w_lstsq_rank(x, rank=4)}

    accuracies = dict()
    name_to_model = dictionary_of_models()

    # name_to_model = {"Ridge": RidgeClassifier()}

    for repr_func_name, repr_func in repr_funcs.items():

        with multiprocessing.Pool() as pool:
            X = np.vstack(map(repr_func, problem.song_analyses))

        if True: # evaluate different models w cv
            accuracies[repr_func_name] = dict()

            for model_name, model in name_to_model.items():
                print(" ", model_name.ljust(50), end="")
                mat = confusion_matrix_for_problem(X, problem.y, model)
                accuracy = accuracy_from_confusion_matrix(mat)
                accuracies[repr_func_name][model_name] = accuracy
                print(f"{accuracy:.4}")

#%% make a single best pipeline

#%%
imshow_confusion_matrix(
    mat,
    [classname.title() for classname in problem.labelnames],
)



#%%

import matplotlib.pyplot as plt
#%%

# Create sequence imshow
chromae_names = ['C', '', 'D', '', 'E', 'F', '', 'G', '', 'A', '', 'B']
sa = [sa for sa in problem.song_analyses if 'Jos2701' in sa.source_fname][0]
plt.figure(figsize=[8, 2], dpi=300)
n_measures = 60
plt.imshow(sa.beats[:n_measures].T, origin='lower')
# plt.xticks(range(0, n_measures, 4), range(1, n_measures + 1, 4))
plt.xticks([], [])
plt.yticks(range(12), chromae_names)
# plt.title('Josquin, A la mort / Monstra te esse matrem')
plt.xlabel('Time (each column one tactus)')
# plt.colorbar()
plt.show()

#%%

# imshow the transition matrix
x = represent_w_lstsq(sa)
plt.figure(figsize=(5, 5), dpi=300)
plt.imshow(x.reshape(12, 12), origin='lower')
plt.title('Transition matrix:\nJosquin, A la mort / Monstra te esse matrem')
plt.xlabel('Activity in predicted tactus')
plt.ylabel('Activity in previous tactus')
plt.xticks(range(12), chromae_names)
plt.yticks(range(12), chromae_names)
plt.show()

print(x.reshape(12, 12))


#%%

from sklearn.decomposition.pca import PCA
from umap import UMAP

transformed = UMAP(n_components=2).fit_transform(X, problem.y)



#%%

plt.figure()
for class_i in set(problem.y):
    # dict(counter.most_common(3)).keys()
    if problem.labelnames[class_i] == 'ano':
        continue
    selector = problem.y == class_i

    if selector.sum() < 20:
        continue
    plt.scatter(
        transformed[selector, 0],
        transformed[selector, 1],
        label=problem.labelnames[class_i],
    )
plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.xticks([])
plt.yticks([])
plt.show()

#%%

plt.figure()
genres = np.array([sa.genre for sa in problem.song_analyses])

for genre in ['Mass', 'Motet', 'Song']:
    # dict(counter.most_common(3)).keys()
    selector = genres == genre
    plt.scatter(
        transformed[selector, 0],
        transformed[selector, 1],
        label=genre,
    )
plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

plt.xticks([])
plt.yticks([])
plt.show()

#%% Confusion matrix

from plotting import imshow_confusion_matrix
from ml import cross_val_predict_confusion_matrix_for_problem

selector = np.array([sa.composer != 'ano' for sa in problem.song_analyses])
X_no_ano = X[selector]
y_no_ano = problem.y[selector]

model = XGBClassifier(n_estimators=1000, n_jobs=-1, max_depth=20, learning_rate=.1)
mat = cross_val_predict_confusion_matrix_for_problem(X_no_ano, y_no_ano, model)
accuracy = accuracy_from_confusion_matrix(mat)

#%%

plt.figure(figsize=(5, 5), dpi=300)
plt.imshow(mat, origin='lower')

labelnames_no_ano = [name for name in problem.labelnames if name != 'ano']
plt.xticks(range(len(labelnames_no_ano)), labelnames_no_ano, rotation=80)
plt.yticks(range(len(labelnames_no_ano)), labelnames_no_ano)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.title('Confusion matrix')
plt.show()


#%% Predict anonymous cases

model.fit(X_no_ano, y_no_ano)
#%%

probs = model.predict_proba(X[np.logical_not(selector)])
predictions = probs.argmax(1)
predictions_probs = probs.max(1)
predictions = [labelnames_no_ano[prediction] for prediction in predictions]

titles = [sa.source_fname.split('/')[-1][:-4] for sa in problem.song_analyses if sa.composer == 'ano']
titles = [title.replace('_', ' ') for title in titles]
titles = ['-'.join(title.split('-')[1:]) for title in titles]

repid_to_fullname = {
    'jos': 'Josquin',
    'rue': 'De La Rue',
    'bus': 'Busnois',
    'com': 'Compere',
    'ock': 'Ockeghem',
    'mar': 'Martini',
    'gas': 'van Weerbeke',
    'duf': 'Du Fay',
    'jap': 'Japart',
    'obr': 'Obrecht',
    'agr': 'Agricola',
    'ort': 'de Orto',
}

pd.DataFrame({
    'JRP id': np.array([sa.details['id'] for sa in problem.song_analyses if sa.composer == 'ano']),
    'Work title': titles,
    'Predicted composer': [repid_to_fullname[prediction] for prediction in predictions],
    'Probability': [f'{p*100:.4}%' for p in predictions_probs]
}).sort_values('Probability', ascending=False).to_excel('out.xlsx')
