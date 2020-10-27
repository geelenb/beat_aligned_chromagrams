from typing import Tuple, List

import tqdm
from xgboost import XGBClassifier

from pitches_problem import PitchesProblem, analysis_from_krn

# Folder that contains jrp-scores;
# git clone https://github.com/josquin-research-project/jrp-scores --recursive
# make update
git_folder_root = "/Users/bgeelen/data/josquin/source_new/jrp-scores"

# wget https://josquin.stanford.edu/cgi-bin/jrp?a=worklist-json -O worklist.json
# json_filename = "/Users/bgeelen/data/josquin/source_new/worklist.json"

import glob
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml import dictionary_of_models, confusion_matrix_for_problem, accuracy_from_confusion_matrix
from representations import represent_w_correlation, represent_w_lstsq, represent_w_lstsq_order, jsymbolic_representer, \
    combined_representation, represent_w_fft, represent_w_welch

#%%


cache_file = "parse_cache.p"

all_genres = {"mass", "motet", "song"}
repid_to_fullname = {
    "ano": "Anonymous",
    "agr": "Agricola",
    "bin": "Binchois",
    "bru": "Brumel",
    "bus": "Busnoys",
    "com": "Compere",
    "das": "Daser",
    "duf": "Du Fay",
    "fva": "Févin",
    "fry": "Frye",
    "gas": "Gaspar",
    "isa": "Isaac",
    "jap": "Japart",
    "jos": "Josquin",
    "rue": "De La Rue",
    "mar": "Martini",
    "mou": "Mouton",
    "obr": "Obrecht",
    "ock": "Ockeghem",
    "ort": "de Orto",
    "pip": "Pipelare",
    "reg": "Regis",
    "tin": "Tinctoris",
}
all_composers = {c.lower() for c in repid_to_fullname.keys()}


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

    analyses = [analysis_from_krn(fname) for fname in tqdm.tqdm(fnames)]

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



#%% Create the 'problem', which contains the state vectors for every work in the dataset


if os.path.exists('problem.p'):
    print('Loading problem from disk cache...')

    with open('problem.p', 'rb') as f:
        problem = pickle.load(f)

else:
    print('Creating new problem by parsing humdrum files...')
    problem = make_problem(
        objective="composer",
        josquin_unsure_is_ano=True,
    )
    # select_composers=['Jos', 'Rue', 'Mar', 'Ock', 'Ano'],

    print('Caching problem to disk...')
    with open('problem.p', 'wb') as f:
        pickle.dump(problem, f)


#%%

repr_funcs = {
    # 'jSymbolic': jsymbolic_representer(),
    # "combined": combined_representation(
    #     lambda x: represent_w_lstsq_order(x, order=1),
    #     jsymbolic_representer()
    # ),
    'welch': lambda x: represent_w_welch(x, n_fft=6),
    "1st order": lambda x: represent_w_lstsq_order(x, order=1),
    # "1st order most_common": lambda x: represent_w_lstsq_order(x, order=1, root_method='most_common'),
    # "1st order endnote": lambda x: represent_w_lstsq_order(x, order=1, root_method='endnote'),
    "fft": lambda x: represent_w_fft(x, n_fft=6),
    # "2nd order": lambda x: represent_w_lstsq_order(x, order=2),
    # "3rd order": lambda x: represent_w_lstsq_order(x, order=3),
    # "4th order": lambda x: represent_w_lstsq_order(x, order=4),
    # "5th order": lambda x: represent_w_lstsq_order(x, order=5),
    # "correlation_1": lambda x: represent_w_correlation(x, order=1),
    # "correlation_4": lambda x: represent_w_correlation(x, order=4),
}

accuracies = dict()
name_to_model = dictionary_of_models()

# name_to_model = {"Ridge": RidgeClassifier()}

ano_i = next(i for i, cls in enumerate(problem.labelnames) if cls == 'ano')
ano_selector = np.array([y == ano_i for y in problem.y])

for repr_func_name, repr_func in repr_funcs.items():
    print(repr_func_name)

    X = np.vstack([repr_func(sa) for sa, is_ano in zip(problem.song_analyses, ano_selector) if not is_ano])
    y = problem.y[np.logical_not(ano_selector)]

    if True: # evaluate different models w cv
        accuracies[repr_func_name] = dict()

        for model_name, model in name_to_model.items():
            print(" ", model_name.ljust(50), end="")
            mat = confusion_matrix_for_problem(X, y, model)
            accuracy = accuracy_from_confusion_matrix(mat)
            accuracies[repr_func_name][model_name] = accuracy
            print(f"{accuracy:.4}")





#%% Write the accuracies of the methods to an .xlsx file
import pandas as pd
(pd.DataFrame(accuracies).round(3) * 100).T.to_excel('accuracies.xlsx')

# !open accuracies.xlsx
#%% Show the most prevalent labels in the dataset

from collections import Counter

prevalences = Counter(
    problem.labelnames[yi]
    for yi in problem.y
)

print(dict(prevalences.most_common(10)))


#%% Details of example to plot

piece_id = 'Ano3225'
piece_author = 'Anonymous'
piece_name = "Helas mon cueur tu m'occiras"

sa = next(sa for sa in problem.song_analyses if piece_id in sa.source_fname)

#%% Print the sequence as a matrix

with np.printoptions(
        suppress=True,
        precision=3,
        floatmode='maxprec',
        sign=' ',
        linewidth=np.inf):
    print(sa.beats[:20, ::-1].T.round(2))

#%% Create sequence imshow

n_measures = 90
chromae_names = ['C', '', 'D', '', 'E', 'F', '', 'G', '', 'A', '', 'B']
plt.figure(figsize=[8, 2.5], dpi=300)
plt.imshow(sa.beats[:n_measures].T, origin='lower', aspect='auto', interpolation='nearest')
# plt.xticks(range(0, n_measures, 4), range(1, n_measures + 1, 4))
plt.xticks([], [])
plt.yticks(range(12), chromae_names)
plt.title(f"{piece_author}, {piece_name}")
plt.xlabel('Time (each column one tactus)')
# plt.colorbar()
plt.show()

#%% Print the transition matrix

x = represent_w_lstsq(sa).reshape(12, 12)

with np.printoptions(suppress=True, precision=3, floatmode='maxprec', sign=' ', linewidth=np.inf):
    print(x.round(3).T[:, ::-1])

#%% Imshow the transition matrix

plt.figure(figsize=(5, 5), dpi=300)
plt.imshow(x[::-1].T, origin='lower')
plt.title(f"Transition matrix:\n{piece_name}")
plt.xlabel('Activity in previous tactus')
plt.ylabel('Activity in predicted tactus')
plt.xticks(range(12), chromae_names[::-1])
plt.yticks(range(12), chromae_names)
plt.colorbar()
plt.show()

#%% Dimensionality reduction

# X = np.vstack(map(lambda x: represent_w_lstsq_order(x, order=1), problem.song_analyses))
X = np.vstack(map(jsymbolic_representer(), problem.song_analyses))

#%%

from sklearn.decomposition.pca import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from umap import UMAP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

for transformed, title in [
    (UMAP(n_components=2, random_state=0).fit_transform(X, problem.y), 'UMAP analysis'),
    # (PCA(2).fit_transform(X), 'PCA analysis'),
    # (NeighborhoodComponentsAnalysis(2).fit_transform(X, problem.y), "NCA")
]:
    plt.figure(figsize=(6, 5), dpi=400)
    for class_i, _ in Counter(problem.y).most_common():
        # dict(counter.most_common(3)).keys()
        selector = problem.y == class_i

        if selector.sum() < 25 or problem.labelnames[class_i] == 'ano':
            plt.scatter(
                transformed[selector, 0],
                transformed[selector, 1],
                c='#00000000'
            )
        else:
            plt.scatter(
                transformed[selector, 0],
                transformed[selector, 1],
                label=repid_to_fullname[problem.labelnames[class_i]],
            )
    plt.gca().legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        fancybox=True,
        shadow=True,
    )
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{title}: by author')
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(6, 5), dpi=400)
    genres = np.array([sa.genre for sa in problem.song_analyses])

    for genre in ['Mass', 'Motet', 'Song']:
        # dict(counter.most_common(3)).keys()
        selector = genres == genre
        plt.scatter(
            transformed[selector, 0],
            transformed[selector, 1],
            label=genre,
        )
    plt.gca().legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    plt.xticks([])
    plt.yticks([])
    plt.title(f'{title}: by genre')
    plt.tight_layout()
    plt.show()

#%% Confusion matrix

from plotting import imshow_confusion_matrix
from ml import cross_val_predict_confusion_matrix_for_problem

ano_selector = np.array([sa.composer != 'ano' for sa in problem.song_analyses])
X_no_ano = X[ano_selector]
y_no_ano = problem.y[ano_selector]

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

# Train on all securely attributed works
model.fit(X_no_ano, y_no_ano)

#%%
# Generate prediction probabilities

probs = model.predict_proba(X[np.logical_not(ano_selector)])
predictions = probs.argmax(1)
predictions_probs = probs.max(1)
predictions = [labelnames_no_ano[prediction] for prediction in predictions]

titles = [sa.source_fname.split('/')[-1][:-4] for sa in problem.song_analyses if sa.composer == 'ano']
titles = [title.replace('_', ' ') for title in titles]
titles = ['-'.join(title.split('-')[1:]) for title in titles]

#%%

most_certain_predictions = pd.DataFrame({
    'JRP id': np.array([sa.details['id'] for sa in problem.song_analyses if sa.composer == 'ano']),
    'Work title': titles,
    'Predicted composer': [repid_to_fullname[prediction] for prediction in predictions],
    'Probability': [f'{p*100:.4}%' for p in predictions_probs]
}).sort_values('Probability', ascending=False)

most_certain_predictions.to_excel('most_certain_predictions.xlsx')

# !open most_certain_predictions.xlsx

#%%

leuven_chansonnier_ids = {'Ano3225', 'Ano3226', 'Ano3229', 'Ano3230', 'Ano3231', 'Ano3232'}
(
    most_certain_predictions
    .loc[
        most_certain_predictions['JRP id'].apply(lambda x: x in leuven_chansonnier_ids)
    ]
    .sort_values('JRP id')
    .to_excel('leuven_chansonnier_predictions.xlsx')
)
# !open leuven_chansonnier_predictions.xlsx


#%% Export all predictions

all_predictions = pd.DataFrame(
    data=probs,
    index=[sa.details['id'] for sa in problem.song_analyses if sa.composer == 'ano'],
    columns=labelnames_no_ano
)
all_predictions['name'] = all_predictions.index.map(
    {sa.details['id'] : sa.source_fname.split('/')[-1][:-4] for sa in problem.song_analyses}.get)
all_predictions.sort_index().to_excel('all_predictions.xlsx')


#%%


predicted_1_step = sa.beats @ represent_w_lstsq(sa).reshape(12, 12)

predicted_2_step = np.hstack([sa.beats[0 + i : i - 2, :] for i in range(2)])\
                       @ (represent_w_lstsq_order(sa, 2).reshape(24, 12))
# predicted_4_step = np.hstack([sa.beats[0 + i : i - 4, :] for i in range(4)])\
#                        @ (represent_w_lstsq_order(sa, 4).reshape(48, 12))

predicted_1_step = np.vstack([np.nan * np.ones((1, 12)), predicted_1_step])
predicted_2_step = np.vstack([np.nan * np.ones((2, 12)), predicted_2_step])
# predicted_4_step = np.vstack([np.nan * np.ones((4, 12)), predicted_4_step])

plt.figure(figsize=(8, 5), dpi=300)
plt.subplot(3, 1, 1)
plt.title('"True" state throughout the song')
plt.imshow(sa.beats[:n_measures].T, origin='lower')
plt.yticks(range(12), chromae_names)
plt.xticks([], [])
# plt.colorbar()

plt.subplot(3, 1, 2)
plt.title('Prediction of the next step when only using the previous state')
plt.imshow(predicted_1_step[:n_measures].T, origin='lower')
plt.yticks(range(12), chromae_names)
plt.xticks([], [])
# plt.colorbar()

plt.subplot(3, 1, 3)
plt.title('Prediction of the next step when using the previous 2 states')
plt.imshow(predicted_2_step[:n_measures].T, origin='lower')
plt.yticks(range(12), chromae_names)
plt.xticks([], [])
# plt.colorbar()



plt.show()



#%%


