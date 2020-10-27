import numpy as np
import scipy.signal
import scipy.linalg
from pitches_problem import SongAnalysis

def mimo_cross_cepstrum(y, windowsize, fs_y):
    """
    y = matrix van signalen (12 x T)
    (oude versie)
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

def represent_w_mimo_cepstrum(
        song_analysis: SongAnalysis,
        nperseg: int = 64,
        nfft: int = 64,
        cutoff: int = 32,
):
    y = song_analysis.beats
    fs, cpsd = scipy.signal.csd(y[:, :, np.newaxis], y[:, np.newaxis, :],
        fs=1.0,
        window='hamming',
        nperseg=nperseg,
        noverlap=None,
        nfft=nfft,
        detrend=False,  # 'constant',
        return_onesided=False,
        scaling='density',
        axis=0,
        average='mean',
    )

    np.nan_to_num(cpsd, copy=False)
    cpsd = np.linalg.det(cpsd)
    np.abs(cpsd, out=cpsd)
    np.log(cpsd, out=cpsd)
    np.nan_to_num(cpsd, copy=False)

    return np.real(np.fft.ifft(cpsd)[1:cutoff]) * np.sqrt(np.arange(1, cutoff))

# def represent_w_mimo_ceptrum(song_analysis: SongAnalysis):
#     matrix = song_analysis.beats
#     return mimo_cross_cepstrum(matrix.T, 128, 1)[1:65]

