import pickle

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from xgboost import XGBClassifier

from representations import represent_w_lstsq_rank, shift_root_to_first_column
import ml
import plotting
from pitches_problem import beats_chroma, analyse_audio, PitchesProblem, SongAnalysis

#%%

life_on_mars_fname = "/Users/bgeelen/Music/iTunes/iTunes Media/Music/Compilations/Life on Mars/02 Life on Mars_.mp3"
# life_on_mars_fname = '/Users/bgeelen/Downloads/Tones and I - Dance Monkey (Lyrics).mp3'
life_on_mars_wav, sr = librosa.load(life_on_mars_fname)
my_funny_valentine_fname = "/Users/bgeelen/Music/iTunes/iTunes Media/Music/Chet Baker/Chet Baker Sings/10 My Funny Valentine.mp3"

#%%
chromae = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
n_octaves = 5
n_bins = 12 * n_octaves + 1
hop_length = 128
sparsity = 0.9
fmin = librosa.note_to_hz("C2")
C = librosa.cqt(
    life_on_mars_wav,
    sr=sr,
    hop_length=hop_length,
    fmin=fmin,
    sparsity=sparsity,
    n_bins=n_bins,
)
abs_C = np.abs(C)
log_C = np.log(abs_C + 0.01)

#%%
h, w = C.shape

plt.figure(figsize=(7.5, 3), dpi=300)
plt.imshow(log_C[:, :10 * sr // hop_length], interpolation="nearest", aspect="auto")
plt.gca().invert_yaxis()

plt.ylabel("Harmonic Frequency")
# plt.title(f"sparsity: {sparsity}")
ytick_locations = list(range(0, h, 12))
ytick_texts = [f"C{2+i}" for i in range(len(ytick_locations))]
plt.yticks(ytick_locations, ytick_texts)

plt.xlabel("Time (s)")
xtick_locations = np.floor(np.arange(0, 11) * sr / hop_length)
xtick_texts = range(11)
plt.xticks(xtick_locations, xtick_texts)

plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

plt.show()


#%%

tempo, beats = librosa.beat.beat_track(
    onset_envelope=librosa.onset.onset_strength(S=abs_C), sr=sr, hop_length=hop_length,
    trim=False
)
print(beats[:2])

#%%

plt.figure(figsize=(7.5, 3), dpi=300)
plt.imshow(log_C[:, :10*sr//hop_length], interpolation="nearest", aspect="auto")
plt.gca().invert_yaxis()

plt.ylabel("Harmonic Frequency")
plt.yticks(ytick_locations, ytick_texts)

plt.xlabel("Time (s)")
plt.xticks(xtick_locations, xtick_texts)

plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

plt.vlines(beats[beats < 10 * sr // hop_length], 0, h - 1, "white")

plt.show()

#%%


plt.figure(figsize=(7.5, 3), dpi=300)
plt.imshow(log_C[:, :10*sr//hop_length], interpolation="nearest", aspect="auto", origin='lower')

plt.ylabel("Harmonic Frequency")
plt.yticks(ytick_locations, ytick_texts)

plt.xlabel("Time (s)")
plt.xticks(xtick_locations, xtick_texts)

plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

plt.hlines(np.arange(0, h, 12), 0, 10*sr//hop_length, "white")

plt.show()

#%%

beat_aligned_chromagram = beats_chroma(C, beats, hop_length)

#%%
n_beats_to_show = 20

plt.figure(figsize=(7.5, 3), dpi=300)
plt.imshow(beat_aligned_chromagram[:n_beats_to_show].T, interpolation="nearest", aspect="auto")
plt.gca().invert_yaxis()
plt.ylabel("Chroma")
plt.yticks(range(12), chromae)
plt.xlabel("Time (Beats)")
plt.xticks(
    range(0, n_beats_to_show, 4),
    np.arange(0, n_beats_to_show, 4) + 1,
)
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))

plt.show()

#%%

sa = analyse_audio(life_on_mars_fname, include_timbres=False)
sa_valentine = analyse_audio(my_funny_valentine_fname, include_timbres=False)

#%%

plt.figure(figsize=(10, 3), dpi=300)
plt.imshow(beat_aligned_chromagram.T, interpolation="nearest", aspect="auto")
plt.gca().invert_yaxis()
plt.ylabel("Chroma")
plt.yticks(range(12), chromae)
plt.xlabel("Measures")
plt.xticks(np.arange(0, len(beat_aligned_chromagram), 16), np.arange(0, len(beat_aligned_chromagram), 16) // 4 + 1)
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))

plt.show()

#%%


def first_order_matrix(beats: np.ndarray):
    x, _, _, _ = np.linalg.lstsq(beats[:-1, :], beats[1:, :])
    return x


x = first_order_matrix(sa.beats)
x_valentine = first_order_matrix(sa_valentine.beats)

#%%

plt.figure(figsize=(4, 4), dpi=300)
plt.imshow(x, origin="upper")

plt.title("Life on Mars - David Bowie")
plt.xlim(-0.5, 11.5)
plt.ylim(-0.5, 11.5)
plt.xticks(range(12), chromae)
plt.yticks(range(12), chromae)
plt.ylabel('"Input" chromae')
plt.xlabel('"Predicted" chromae')
plt.gca().invert_yaxis()

plt.show()

#%%

plt.figure(figsize=(4, 4), dpi=300)
plt.imshow(x_valentine, origin="upper")

plt.title("My Funny Valentine - Chet Baker")

plt.xlim(-0.5, 11.5)
plt.ylim(-0.5, 11.5)
plt.ylabel('"Input" chromae')
plt.xlabel('"Predicted" chromae')
plt.xticks(range(12), chromae)
plt.yticks(range(12), chromae)
plt.gca().invert_yaxis()

plt.show()

#%%


# with open('ismir_genre_dataset.p', 'rb') as f:
#     ismir_genre = pickle.load(f)
with open("gtzan.p", "rb") as f:
    gtzan: PitchesProblem = pickle.load(f)

#%%

xs = np.vstack(
    [first_order_matrix(sa.beats[:, :12]).flatten() for sa in gtzan.song_analyses]
)

#%%

linearmodel = LogisticRegression(fit_intercept=False)
linearmodel.fit(xs, gtzan.y)


#%%

plt.figure(figsize=(4, 4), dpi=300)
plt.imshow(
    linearmodel.coef_[gtzan.labelnames.index("jazz")].reshape(12, 12), origin="upper"
)

plt.title('"Jazz" transitions')

plt.xlim(-0.5, 11.5)
plt.ylim(-0.5, 11.5)
plt.ylabel('"Input" chromae')
plt.xlabel('"Predicted" chromae')
plt.xticks(range(12), chromae)
plt.yticks(range(12), chromae)
plt.gca().invert_yaxis()

plt.show()

#%%


plt.figure(figsize=(4, 4), dpi=300)
plt.imshow(
    linearmodel.coef_[gtzan.labelnames.index("rock")].reshape(12, 12), origin="upper"
)

plt.title('"Rock" transitions')

plt.xlim(-0.5, 11.5)
plt.ylim(-0.5, 11.5)
plt.ylabel('"Input" chromae')
plt.xlabel('"Predicted" chromae')
plt.xticks(range(12), chromae)
plt.yticks(range(12), chromae)
plt.gca().invert_yaxis()

plt.show()

#%%

model = XGBClassifier(n_estimators=1000)
predictions = cross_val_predict(model, xs, gtzan.y, cv=5)

#%%

mat = confusion_matrix(gtzan.y, predictions)
accuracy = ml.accuracy_from_confusion_matrix(mat)
plotting.imshow_confusion_matrix(
    mat, gtzan.labelnames, f"GTZAN cross validation accuracy: {accuracy * 100:2.1f}%"
)


#%%

def fourth_order_lstsq(song_analysis):
    # matrix = shift_root_to_first_column(song_analysis.beats)
    matrix = song_analysis.beats[:, :12]
    X = np.hstack([matrix[0 + i: i - 4, :] for i in range(4)])
    Y = matrix[4:, :]
    representation, _, _, _ = np.linalg.lstsq(X, Y)
    return representation.flatten()


xs_lstsq_2 = np.vstack([fourth_order_lstsq(sa) for sa in gtzan.song_analyses])

#%%=
model = XGBClassifier(n_estimators=1000)
predictions = cross_val_predict(model, xs_lstsq_2, gtzan.y, cv=5)

#%%


mat = confusion_matrix(gtzan.y, predictions)
accuracy = ml.accuracy_from_confusion_matrix(mat)
plotting.imshow_confusion_matrix(
    mat, gtzan.labelnames, f"GTZAN cross validation accuracy: {accuracy * 100:2.1f}%"
)


#%%

def represent_w_correlation(sa: SongAnalysis, rank: int = 1):
    matrix = sa.beats[:, :12]
    if rank == 0:
        representation = matrix.T @ matrix
        # representation = np.cov(matrix.T)
    else:
        representation = matrix[:-rank].T @ matrix[rank:]
        # representation = np.cov(matrix[:-rank].T, matrix[rank:].T)[12:, :12]
    return representation.flatten()

corr_bowie = represent_w_correlation(sa)
corr_valentine = represent_w_correlation(sa_valentine)

#%%


plt.figure(figsize=(4, 4), dpi=300)
plt.imshow(
    corr_bowie.reshape(12, 12), origin="upper"
)

plt.title('Chroma covariances: Life on Mars ')

plt.xlim(-0.5, 11.5)
plt.ylim(-0.5, 11.5)
plt.ylabel('"Input" chromae')
plt.xlabel('"Predicted" chromae')
plt.xticks(range(12), chromae)
plt.yticks(range(12), chromae)
plt.gca().invert_yaxis()

plt.show()

#%%



plt.figure(figsize=(4, 4), dpi=300)
plt.imshow(
    corr_valentine.reshape(12, 12), origin="upper"
)

plt.title('Chroma covariances: My Funny Valentine')

plt.xlim(-0.5, 11.5)
plt.ylim(-0.5, 11.5)
plt.ylabel('"Input" chromae')
plt.xlabel('"Predicted" chromae')
plt.xticks(range(12), chromae)
plt.yticks(range(12), chromae)
plt.gca().invert_yaxis()

plt.show()

#%%
shifter = np.roll(np.arange(0, 7*12, 7) % 12, 6)

#%%

n_beats_to_show = 30
plt.figure(figsize=(4, 5), dpi=300)
plt.imshow(beat_aligned_chromagram[:n_beats_to_show], interpolation="nearest", aspect="auto")

plt.xlabel("Chroma")
plt.xticks(range(12), chromae)
plt.ylabel("Time (Beats)")
plt.yticks(
    range(0, len(beat_aligned_chromagram[:n_beats_to_show]), 4),
    np.arange(0, len(beat_aligned_chromagram[:n_beats_to_show]), 4) + 1,
)
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
# plt.xlim(-.5, len())

plt.show()

#%%

chromae_roman = ['I', '', 'II', '', 'III', 'IV', '', 'V', '', 'VI', '', 'VII']

shift = -(beat_aligned_chromagram.sum(0).argmax())
n_beats_to_show = 30
plt.figure(figsize=(4, 5), dpi=300)
plt.imshow(np.roll(beat_aligned_chromagram[:n_beats_to_show], shift, axis=1), interpolation="nearest", aspect="auto")

plt.xlabel("Chroma")
plt.xticks(range(12), chromae_roman)
plt.ylabel("Time (Beats)")
plt.yticks(
    range(0, len(beat_aligned_chromagram[:n_beats_to_show]), 4),
    np.arange(0, len(beat_aligned_chromagram[:n_beats_to_show]), 4) + 1,
)
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
# plt.xlim(-.5, len())

plt.show()

#%%

plt.figure(figsize=(4, 4), dpi=300)
plt.imshow(
    corr_valentine.reshape(12, 12), origin="upper"
)

plt.title('Chroma covariances: My Funny Valentine')

plt.xlim(-0.5, 11.5)
plt.ylim(-0.5, 11.5)
plt.ylabel('"Input" chromae')
plt.xlabel('"Predicted" chromae')
plt.xticks(range(12), chromae)
plt.yticks(range(12), chromae)
plt.gca().invert_yaxis()

plt.show()

#%%

plt.figure(figsize=(4, 4), dpi=300)
plt.imshow(
    corr_valentine.reshape(12, 12)[shifter][:, shifter], origin="upper"
)

plt.title('Chroma covariances: My Funny Valentine')

plt.xlim(-0.5, 11.5)
plt.ylim(-0.5, 11.5)
plt.ylabel('"Input" chromae')
plt.xlabel('"Predicted" chromae')
plt.xticks(range(12), np.array(chromae)[shifter][:, shifter])
plt.yticks(range(12), np.array(chromae)[shifter][:, shifter])
plt.gca().invert_yaxis()

plt.show()

#%%

xs = np.vstack(
    represent_w_correlation(sa, rank=1) for sa in gtzan.song_analyses
)

model = XGBClassifier(n_estimators=1000)
predictions = cross_val_predict(model, xs, gtzan.y, cv=5)

mat = confusion_matrix(gtzan.y, predictions)
accuracy = ml.accuracy_from_confusion_matrix(mat)
plotting.imshow_confusion_matrix(
    mat, gtzan.labelnames, f"GTZAN cross validation accuracy: {accuracy * 100:2.1f}%"
)

#%%

xs = np.vstack(
    represent_w_correlation(sa, rank=4) for sa in gtzan.song_analyses
)

model = XGBClassifier(n_estimators=1000)
predictions = cross_val_predict(model, xs, gtzan.y, cv=5, n_jobs=-1)

mat = confusion_matrix(gtzan.y, predictions)
accuracy = ml.accuracy_from_confusion_matrix(mat)
plotting.imshow_confusion_matrix(
    mat, gtzan.labelnames, f"GTZAN cross validation accuracy: {accuracy * 100:2.1f}%"
)

#%%

def represent_w_correlation_shifted(sa: SongAnalysis, rank: int = 1):
    matrix = shift_root_to_first_column(sa.beats[:, :12])
    if rank == 0:
        representation = matrix.T @ matrix
        # representation = np.cov(matrix.T)
    else:
        representation = matrix[:-rank].T @ matrix[rank:]
        # representation = np.cov(matrix[:-rank].T, matrix[rank:].T)[12:, :12]
    return representation.flatten()

xs = np.vstack(
    represent_w_correlation_shifted(sa, rank=4) for sa in gtzan.song_analyses
)

model = XGBClassifier(n_estimators=1000)
predictions = cross_val_predict(model, xs, gtzan.y, cv=5, n_jobs=-1)

mat = confusion_matrix(gtzan.y, predictions)
accuracy = ml.accuracy_from_confusion_matrix(mat)
plotting.imshow_confusion_matrix(
    mat, gtzan.labelnames, f"GTZAN cross validation accuracy: {accuracy * 100:2.1f}%"
)

#%%

def represent_w_correlation_shifted_pitches(sa: SongAnalysis, rank: int = 1):
    matrix = shift_root_to_first_column(sa.beats)
    if rank == 0:
        representation = matrix.T @ matrix
        # representation = np.cov(matrix.T)
    else:
        representation = matrix[:-rank].T @ matrix[rank:]
        # representation = np.cov(matrix[:-rank].T, matrix[rank:].T)[12:, :12]
    return representation.flatten()

xs = np.vstack(
    represent_w_correlation_shifted_pitches(sa, rank=4) for sa in gtzan.song_analyses
)

model = XGBClassifier(n_estimators=1000)
predictions = cross_val_predict(model, xs, gtzan.y, cv=5, n_jobs=-1)

mat = confusion_matrix(gtzan.y, predictions)
accuracy = ml.accuracy_from_confusion_matrix(mat)
plotting.imshow_confusion_matrix(
    mat, gtzan.labelnames, f"GTZAN cross validation accuracy: {accuracy * 100:2.1f}%"
)

#%%


def plot_umap(X, y, labelnames):
    X = StandardScaler().fit_transform(X)

    X = UMAP(n_components=2).fit_transform(X, y)

    plt.figure(figsize=(8, 5), dpi=300)

    for i in set(y):
        artist_name = labelnames[i]
        x = X[y == i]
        plt.scatter(*x.T, label=artist_name)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.legend()
    plt.show()


plot_umap(xs, gtzan.y, gtzan.labelnames)
