from functools import partial

import numpy as np
import scipy
import scipy.linalg
import scipy.signal

from pitches_problem import SongAnalysis

np.warnings.filterwarnings("ignore")


def array_of_dicts_to_dict_of_arrays(list_of_dicts):
    return {
        key: np.array([item[key] for item in list_of_dicts]) for key in list_of_dicts[0]
    }


def shift_root_to_first_column(pitches: np.ndarray):
    root = pitches[:, :12].sum(0).argmax()
    return np.hstack([pitches[:, root:12], pitches[:, :root], pitches[:, 12:]])


def audio_analysis_to_pitches(audio_analysis):
    pitches = array_of_dicts_to_dict_of_arrays(audio_analysis["segments"])["pitches"]
    return shift_root_to_first_column(pitches)


def audio_analysis_to_pitches_and_timbres(audio_analysis):
    segments = array_of_dicts_to_dict_of_arrays(audio_analysis["segments"])
    return np.hstack(
        (shift_root_to_first_column(segments["timbre"]), segments["pitches"])
    )



def represent_w_lstsq(song_analysis: SongAnalysis):
    # matrix = shift_root_to_first_column(song_analysis.beats)
    matrix = song_analysis.beats
    representation, _, _, _ = np.linalg.lstsq(matrix[:-1, :], matrix[1:, :])
    return representation.flatten()


def represent_w_lstsq_1(song_analysis: SongAnalysis):
    matrix = shift_root_to_first_column(song_analysis.beats)
    X = np.hstack([matrix[:-1, :], np.ones([len(matrix) - 1, 1])])
    Y = matrix[1:, :]
    representation, _, _, _ = np.linalg.lstsq(X, Y)
    return representation.flatten()


def null_space(A, r):
    U, s, V = np.linalg.svd(A, 0)
    return V[-r:, :].T


def represent_w_tls(song_analysis: SongAnalysis):
    matrix = shift_root_to_first_column(song_analysis.beats)
    # X = np.hstack([matrix[:-1, :], np.ones([len(matrix), 1])])
    X = matrix[:-1, :]
    Y = matrix[1:, :]
    XY = np.hstack([X, Y])
    representation = null_space(XY, matrix.shape[1])

    # representation, _, _, _ = np.linalg.lstsq(X, Y)
    return representation.flatten()


def represent_w_lstsq_eigvecs(song_analysis: SongAnalysis):
    matrix = shift_root_to_first_column(song_analysis.beats)
    M, _, _, _ = np.linalg.lstsq(matrix[:-1, :], matrix[1:, :])
    representation = np.linalg.eig(M)[1]
    representation = np.vstack((representation.real, representation.imag))
    return representation.flatten()


def represent_w_lstsq_eigvecs_and_vals(song_analysis: SongAnalysis):
    matrix = shift_root_to_first_column(song_analysis.beats)
    M, _, _, _ = np.linalg.lstsq(matrix[:-1, :], matrix[1:, :])
    vals, vecs = np.linalg.eig(M)
    representation = np.vstack((vals.real, vals.imag, vecs.real, vecs.imag))
    return representation.flatten()


def represent_w_lstsq_lookback(song_analysis: SongAnalysis, rank=4):
    matrix = shift_root_to_first_column(song_analysis.beats)
    X = matrix[:-rank, :]
    Y = matrix[rank:, :]
    representation, _, _, _ = np.linalg.lstsq(X, Y)
    return representation.flatten()


def represent_w_lstsq_rank(song_analysis: SongAnalysis, rank=4):
    # matrix = shift_root_to_first_column(song_analysis.beats)
    matrix = song_analysis.beats
    X = np.hstack([matrix[0 + i : i - rank, :] for i in range(rank)])
    Y = matrix[rank:, :]
    representation, _, _, _ = np.linalg.lstsq(X, Y)
    return representation.flatten()


def represent_w_hankel_svd(song_analysis: SongAnalysis, rank=20):
    matrix = shift_root_to_first_column(song_analysis.beats)
    X = np.hstack([matrix[i : i - rank] for i in range(rank)])  # hankelize
    # Y = matrix[rank:, :]

    U, s, V = np.linalg.svd(X)
    U = U[:, :rank]
    U = U @ np.diag(np.sqrt(s[:rank]))
    representation, _, _, _ = np.linalg.lstsq(U[:-1, :], U[1:, :])
    return representation.flatten()


def represent_w_hankel_svd_eig(song_analysis: SongAnalysis, rank=20):
    matrix = shift_root_to_first_column(song_analysis.beats)
    X = np.hstack([matrix[i : i - rank] for i in range(rank)])  # hankelize

    U, s, V = np.linalg.svd(X)
    U = U[:, :rank]
    U = U @ np.diag(np.sqrt(s[:rank]))
    A, _, _, _ = np.linalg.lstsq(U[:-1, :], U[1:, :])
    representation = np.linalg.eigvals(A)
    representation = np.vstack((representation.real, representation.imag))
    return representation.flatten()


# def represent_w_eigvecs(song_analysis: SongAnalysis):
#     matrix = song_analysis.beats
#     representation, _, _, _ = np.linalg.lstsq(matrix[:-1], matrix[1:])
#     np.linalg.eigv(representation)
#     return representation.flatten()


def represent_w_lstsq_diff(song_analysis: SongAnalysis):
    matrix = shift_root_to_first_column(song_analysis.beats)
    representation, _, _, _ = np.linalg.lstsq(
        matrix[:-1, :], matrix[1:, :] - matrix[:-1, :]
    )
    return representation.flatten()


def represent_w_mean(song_analysis: SongAnalysis):
    matrix = song_analysis.beats
    return matrix.mean(0)

def represent_w_correlation_plain(song_analysis: SongAnalysis, rank=1):
    matrix=song_analysis.beats
    if rank == 1:
        representation = matrix.T @ matrix
    else:
        representation = matrix[:-rank].T @ matrix[rank:]
    return representation.flatten()

def represent_w_correlation(song_analysis: SongAnalysis, rank=1):
    matrix = shift_root_to_first_column(song_analysis.beats)
    if rank == 0:
        representation = matrix.T @ matrix
    else:
        representation = matrix[:-rank].T @ matrix[rank:]
    return representation.flatten()


def represent_w_dct(song_analysis: SongAnalysis):
    matrix = song_analysis.beats
    zero_rows_to_add = 5 - matrix.shape[0]
    if zero_rows_to_add > 0:
        matrix = np.pad(matrix, [(0, zero_rows_to_add), (0, 0)])
    dcted = scipy.fftpack.dct(matrix, axis=0)
    dcted = np.vstack((dcted[:3, :], dcted[-5:, :]))
    return dcted.flatten()


def represent_w_diagonals_of_lstsq(song_analysis: SongAnalysis):
    matrix = song_analysis.beats
    M, _, _, _ = np.linalg.lstsq(matrix[:-1, :], matrix[1:, :] - matrix[:-1, :])
    l = M.shape[0]
    M = np.hstack((M, M))

    return np.vstack(np.diag(M, i) for i in range(l)).sum(1)

def represent_w_bpm(song_analysis: SongAnalysis):
    return np.array([song_analysis.bpm])

def represent_w_num_beats(song_analysis: SongAnalysis):
    return np.array([len(song_analysis.beats)])

def represent_w_song_length(song_analysis: SongAnalysis):
    return np.array([song_analysis.duration])

def combined_representation(*repr_functions):
    def combined_representation_f(song_analysis):
        return np.concatenate([f(song_analysis) for f in repr_functions])

    return combined_representation_f


def get_all_repr_functions():
    fs = []
    for varname in globals():
        value = eval(varname)
        if callable(value) and varname.startswith("represent_w_"):
            fs.append(value)

    return fs


def get_best_repr_function():
    return combined_representation(
        represent_w_bpm,
        represent_w_song_length,
        represent_w_num_beats,
        represent_w_mean,
        partial(represent_w_correlation, rank=0),
        represent_w_correlation,
        partial(represent_w_correlation, rank=2),
        partial(represent_w_correlation, rank=4),
    )


if __name__ == '__main__':
    f = combined_representation(represent_w_mean, represent_w_dct)
    print(f.__name__)