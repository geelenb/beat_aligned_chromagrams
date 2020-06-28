import numpy as np
import scipy.signal
import scipy.linalg
from pitches_problem import SongAnalysis

def mimo_cross_cepstrum(y, windowsize, fs_y):
    """
    y = matrix van signalen (12 x T)
    windowsize = 2**(7)
    check docs van csd
    """
    _, P_array = scipy.signal.csd(
        np.repeat(y, y.shape[0], axis=0),
        np.tile(y, [y.shape[0], 1]),
        fs=fs_y,
        nperseg=windowsize,
        nfft=2 ** 11,
        return_onesided=True,
    )
    P_matrix = np.empty((y.shape[0], y.shape[0], P_array.shape[1]), dtype=complex)
    Pyy = np.empty((P_array.shape[1]), dtype=complex)

    for i in range(P_array.shape[0]):
        P_matrix[i // y.shape[0], i % y.shape[0]] = P_array[i, :]

    for i in range(P_array.shape[1]):
        if False:
            eig_values = scipy.linalg.eig(
                P_matrix[:, :, i], overwrite_a=True, check_finite=False
            )[0]
            Pyy[i] = np.product(eig_values[np.abs(eig_values) > 0])
        else:
            Pyy[i] = scipy.linalg.det(P_matrix[:, :, i])

    power_cepstrum = np.abs(
        np.fft.irfft(np.log(Pyy, out=np.zeros_like(Pyy), where=(Pyy != 0)))
    )

    return power_cepstrum


def represent_w_mimo_ceptrum(song_analysis: SongAnalysis):
    matrix = song_analysis.beats
    return mimo_cross_cepstrum(matrix.T, 128, 1)[1:65]

