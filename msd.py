import glob
import os
from collections import namedtuple
from multiprocessing.pool import Pool
import time

import h5py
import numpy as np
from tqdm import tqdm

# TODO: Dataclass?
from pitches_problem import SongAnalysis

# Song = namedtuple(
#     "Song",
#     [
#         "analyzer_version",
#         "artist_7digitalid",
#         "artist_familiarity",
#         "artist_hottnesss",
#         "artist_id",
#         "artist_latitude",
#         "artist_location",
#         "artist_longitude",
#         "artist_mbid",
#         "artist_name",
#         "artist_playmeid",
#         "genre",
#         "idx_artist_terms",
#         "idx_similar_artists",
#         "release",
#         "release_7digitalid",
#         "song_hotttnesss",
#         "song_id",
#         "title",
#         "track_7digitalid",
#     ],
# )

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

if __name__ == '__main__':

    # single threaded:
    fnames = glob.glob(os.path.join(msd_data_root, "**", "*.h5"), recursive=True)
    fname = fnames[0]
    sa = analysis_from_h5(fname)

    # results = [analysis_from_h5(fname) for fname in tqdm(fnames)]
    # ~50 files per second
    # 10_000 files = 200 seconds
    # 1_000_000 files = 20_000 seconds
