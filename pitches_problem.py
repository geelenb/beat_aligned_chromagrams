from collections import namedtuple
from dataclasses import dataclass, field

import h5py
import librosa
import numpy as np
from scipy.fftpack import dct
from typing import List

sr = None
hops_per_s = 20



@dataclass
class SongAnalysis:
    beats: np.ndarray
    bpm: float = np.nan
    duration: float = np.nan
    source_fname: str = ""
    details: dict = field(default_factory=dict)
    genre: str = ""
    composer: str = ""

@dataclass
class PitchesProblem:
    song_analyses: List[SongAnalysis]
    y: np.ndarray
    labelnames: List[str]

def beats_timbres(C, beats):
    log_C = np.log(0.001 + C)
    cqt_segments = (log_C[:, beats[i] : beats[i + 1]] for i in range(len(beats) - 1))

    beats_2d_dcts = (dct(dct(segment, axis=0), axis=1) for segment in cqt_segments)
    return np.vstack(
        [
            (
                beat_dct[0, 0],  # 1
                beat_dct[1, 0],  # 2
                beat_dct[2, 0],  # 3
                beat_dct[0, 1],  # 4
                beat_dct[3, 0],  # 5
                beat_dct[0, 2],  # 6
                beat_dct[4, 0],  # 7
                beat_dct[1, 1],  # 8
                beat_dct[5, 0],  # 9
                beat_dct[1, 2],  # 10
                beat_dct[0, 3],  # 11
                beat_dct[6, 0],  # 12
            )
            for beat_dct in beats_2d_dcts
        ]
    )


def analyse_audio(fname, sr=None, include_timbres=True) -> SongAnalysis:
    wav, sr = librosa.load(fname, sr)
    hop = 2 ** int(np.log2(sr / hops_per_s))

    C = np.abs(
        librosa.cqt(wav, sr=sr, hop_length=hop, bins_per_octave=12, sparsity=0.5)
    )

    onset_envelope = librosa.onset.onset_strength(S=C)
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_envelope, sr=sr, hop_length=hop
    )

    if include_timbres:
        beats = np.hstack([beats_chroma(C, beats, hop), beats_timbres(C, beats)])
    else:
        beats = beats_chroma(C, beats, hop)

    return SongAnalysis(
        beats, tempo, len(wav) / sr, source_fname=fname, details={"sr": sr , "beat_track": beats}
    )


def beats_chroma(C, beats, hop_length):
    chromagram_librosa = np.abs(
        librosa.feature.chroma_cqt(C=C, sr=sr, hop_length=hop_length)
    )
    chroma_segments = [
        chromagram_librosa[:, beats[i] : beats[i + 1]] for i in range(len(beats) - 1)
    ]
    beats_chroma = np.vstack([np.mean(segment, axis=1) for segment in chroma_segments])
    beats_chroma = np.nan_to_num(beats_chroma, copy=False)

    # normalize between 0 and 1
    beats_chroma /= beats_chroma.max(1, keepdims=True)
    beats_chroma = np.nan_to_num(beats_chroma, copy=False)
    beats_chroma = np.square(beats_chroma, out=beats_chroma)

    return beats_chroma


def analysis_from_h5py(fname, include_timbres=False) -> SongAnalysis:
    with h5py.File(fname, "r") as f:
        # metadata = SongAnalysis._make(f["metadata"]["songs"][0])

        pitches = f["analysis"]["segments_pitches"]

        if include_timbres:
            pitches = np.hstack([pitches, f["analysis"]["segments_timbre"],])

        return SongAnalysis(
            pitches,
            f["analysis"]["songs"][0]["tempo"],
            f["analysis"]["songs"][0]["duration"],
            dict(),
        )



if __name__ == "__main__":
    if False:
        sa = analyse_audio(
            "/Users/bgeelen/Music/iTunes/iTunes Media/Music/Compilations/Life on Mars/02 Life on Mars_.mp3",
            include_timbres=True,
        )

        # sb = analysis_from_h5py('/Users/bgeelen/Data/msd/data/a/a/a/TRAAAAW128F429D538.h5', True)
        f = h5py.File("/Users/bgeelen/Data/msd/data/a/a/a/TRAAAAW128F429D538.h5")

    SongAnalysis(np.ones([1]), 120.9, 180.0)
