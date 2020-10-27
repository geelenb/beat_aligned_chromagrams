import glob
import os
import pickle
from collections import namedtuple
from multiprocessing.pool import Pool
import time

import h5py
import numpy as np
from tqdm import tqdm
from oliver import represent_w_mimo_cepstrum

from pitches_problem import SongAnalysis

#%%
from representations import represent_w_welch, represent_w_lstsq_order

msd_root = "/Users/bgeelen/Data/msd"
msd_data_root = os.path.join(msd_root, "data")


def get_dataset_colnames(dataset):
    i_to_colname = {
        int(k.split("_")[1]): v
        for k, v in dataset.attrs.items()
        if k.startswith("FIELD") and k.endswith("_NAME")
    }

    return [i_to_colname[i] for i in i_to_colname]



keys_to_retain = {
    'analysis_sample_rate',
    'analyzer_version',
    'artist_familiarity',
    'artist_hotttnesss',
    'artist_latitude',
    'artist_location',
    'artist_longitude',
    'artist_name',
    'artist_terms',
    'danceability',
    'duration',
    'energy',
    'genre',
    'key',
    'loudness',
    'mode',
    'release',
    'song_hotttnesss',
    'song_id',
    'tempo',
    'time_signature',
    'title',
    'track_id',
}

def analysis_from_h5(fname) -> SongAnalysis:
    with h5py.File(fname, "r") as f:
        analysis = f['analysis']['songs']
        metadata = f['metadata']['songs']

        details = {
            **dict(zip(analysis.dtype.names, analysis[0])),
            **dict(zip(metadata.dtype.names, metadata[0])),
            "artist_terms": list(f["metadata"]["artist_terms"]),
        }
        details = {k: details[k] for k in keys_to_retain}

        return SongAnalysis(
            beats=np.hstack([f["analysis"]["segments_pitches"], f["analysis"]["segments_timbre"]]),
            bpm=details['tempo'],
            duration=details['duration'],
            source_fname=fname,
            details=details,
            genre=details['genre'],
            composer=details['artist_name']
        )


#%%

if __name__ == '__main__':
    # single threaded:
    fnames = glob.glob(os.path.join(msd_data_root, "**", "*.h5"), recursive=True)
    fname = fnames[0]
    sa = analysis_from_h5(fname)

    cache_fname = 'h5_analyses.p'
    if os.path.exists(cache_fname):
        with open(cache_fname, 'rb') as f:
            analyses = pickle.load(f)
    else:
        analyses = [analysis_from_h5(fname) for fname in tqdm(fnames)]

        with open(cache_fname, 'wb') as f:
            pickle.dump(analyses, f)
    # ~50 files per second
    # 10_000 files = 200 seconds
    # 1_000_000 files = 20_000 seconds

    #%%
    representations = [
        represent_w_lstsq_order(sa)
        for sa in tqdm(analyses)
    ]

    representations = np.nan_to_num(representations)
    #%%
    # import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, confusion_matrix


    #%%
    if False:
        clusterer = KMeans(10)
        clusterer.fit(np.array(representations))
        labels = clusterer.labels_

    #%%
    import collections, itertools

    counter = collections.Counter(
        itertools.chain.from_iterable(
            sa.details['artist_terms'] for sa in analyses
        )
    )
    counter.most_common(10)


    #%%

    # import sklearn.model_selection
    import sklearn, sklearn.ensemble
    from xgboost.sklearn import XGBClassifier

    for term, prevalence in counter.most_common(10):
        acc = sklearn.model_selection.cross_val_score(
            XGBClassifier(),
            representations,
            np.array([term in sa.details['artist_terms'] for sa in analyses]),
            scoring='roc_auc',
        ).mean()

        print(
            str(term).rjust(20),
            str(prevalence).rjust(10),
            acc
        )


