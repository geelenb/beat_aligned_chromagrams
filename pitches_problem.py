import os
from collections import namedtuple
from dataclasses import dataclass, field

import h5py
import librosa
import numpy as np
from scipy.fftpack import dct
from typing import List
import music21
from music21 import midi
import pandas as pd

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


def _attribution_levels_for_work(work):  # -> dict[str, Tuple[int, str]]
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

def analysis_from_krn(fname: str) -> SongAnalysis:
    work = music21.converter.parse(fname)
    id = os.path.split(fname)[1].split("-")[0]

    notes = [x for x in work.recurse() if isinstance(x, music21.note.Note)]
    first_measure = work.measures(0, 1)[0]
    while not isinstance(first_measure, music21.stream.Measure):
        first_measure = first_measure.measures(0, 1)[0]
    measure_length = first_measure.quarterLength

    if (measure_length % 3.0) == 0.0:
        tactus_length = measure_length / 3
    else:
        tactus_length = measure_length / 2

    X = np.zeros((int(work.quarterLength / tactus_length), 12))
    for note in notes:
        quarterLength = note.quarterLength
        offset = note.getOffsetInHierarchy(work)
        X[
            int(offset // tactus_length) : int((offset + quarterLength) / tactus_length),
            note.pitch.midi % 12,
        ] += min(1.0, quarterLength / tactus_length)

        if (offset + quarterLength) % tactus_length > 0.0:
            # if the note ends before the end of a tactus
            try:
                X[
                    int((offset + quarterLength) / tactus_length),
                    note.pitch.midi % 12
                ] += ((offset + quarterLength) % tactus_length) / tactus_length
            except IndexError:
                pass

        if offset % tactus_length > 0.0:
            # if the note starts after the start of a tactus
            try:
                X[
                    int(offset / tactus_length),
                    note.pitch.midi % 12
                ] -= offset % tactus_length / tactus_length
            except IndexError:
                pass

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
            "attributions": _attribution_levels_for_work(work),
        },
        genre=genre,
        composer=composer,
    )

def analysis_from_midi(midi_fname, h5_fname=None) -> SongAnalysis:
    midi.MidiFile(midi_fname)


    if h5_fname:
        pass


if __name__ == "__main__":
    if False:
        sa = analyse_audio(
            "/Users/bgeelen/Music/iTunes/iTunes Media/Music/Compilations/Life on Mars/02 Life on Mars_.mp3",
            include_timbres=True,
        )

        # sb = analysis_from_h5py('/Users/bgeelen/Data/msd/data/a/a/a/TRAAAAW128F429D538.h5', True)
        f = h5py.File("/Users/bgeelen/Data/msd/data/a/a/a/TRAAAAW128F429D538.h5")

    if True:
        fname = '/Users/bgeelen/Data/lakh/lmd_matched/A/A/A/TRAAAGR128F425B14B/1d9d16a9da90c090809c153754823c2b.mid'
        midifile = midi.MidiFile()
        midifile.open(fname)
        midifile.read()
        midifile.close()

        stream = music21.midi.translate.midiFileToStream(midifile)
        notes = [x for x in work.recurse() if isinstance(x, music21.note.Note)]
        first_measure = work.measures(0, 1)[0]
        measure_length = first_measure.quarterLength





