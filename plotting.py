import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patheffects as path_effects
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from mpl_toolkits import mplot3d
import numpy as np

from pitches_problem import PitchesProblem


def plot_pca(X, y, labelnames):
    X = StandardScaler().fit_transform(X)
    X = PCA(2).fit_transform(X, y)

    plt.figure()

    for i in set(y):
        name = labelnames[i]
        x = X[y == i]
        plt.scatter(*x.T, label=name)

    plt.legend()
    plt.show()

def plot_umap(X, y, labelnames):
    X = StandardScaler().fit_transform(X)

    X = UMAP(n_components=2).fit_transform(X, y)

    plt.figure()

    for i in set(y):
        artist_name = labelnames[i]
        x = X[y == i]
        plt.scatter(*x.T, label=artist_name)

    plt.legend()
    plt.show()


def plot_pca_3d(X, y, labelnames):
    X = StandardScaler().fit_transform(X)
    pca = PCA(3)
    X = pca.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in set(y):
        name = labelnames[i]
        x = X[y == i]
        plt.scatter(*x.T, label=name)

    plt.legend()
    plt.show()


def plot_umap_3d(X, y, labelnames):
    X = StandardScaler().fit_transform(X)
    X = UMAP(n_components=3).fit_transform(X, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in set(y):
        artist_name = labelnames[i]
        x = X[y == i]
        plt.scatter(*x.T, label=artist_name)

    plt.legend()
    plt.show()


def imshow_chroma(chroma, title="", fname=None):
    plt.figure(figsize=(30, 2))
    plt.imshow(chroma.T, aspect="auto", interpolation="nearest")
    # plt.colorbar()
    plt.title(title)
    plt.gca().invert_yaxis()

    if fname:
        plt.savefig(fname)

    plt.show()


def imshow_confusion_matrix(mat: np.ndarray, classnames, title=None, out_fname=None):
    l = len(mat)
    plt.figure(figsize=(.55 * len(mat), .5*len(mat)), dpi=700)
    plt.imshow(mat / np.sum(mat, 1))
    if title:
        plt.title(title)
    plt.xticks(range(l), [s[:4] for s in classnames])
    plt.yticks(range(l), classnames)
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    # plt.colorbar()
    for i in range(l):
        for j in range(l):
            plt.text(
                j,
                i,
                mat[i, j],
                color="white",
                horizontalalignment="center",
                verticalalignment="center",
                path_effects=[
                    path_effects.Stroke(linewidth=2, foreground="black"),
                    path_effects.Normal(),
                ],
            )
    plt.xlim(-0.5, l - 0.5)
    plt.ylim(l - 0.5, -0.5)

    plt.tight_layout()
    if out_fname is None:
        plt.show()
    else:
        plt.savefig(out_fname, transparent=True, bbox_inches='tight', dpi='figure')


shifter = np.roll(np.arange(0, 7*12, 7) % 12, 6)

def imshow_transition_matrix(vector_144: np.ndarray, shift: bool=True):
    M = vector_144.reshape(12, 12)
    chromae = np.array(['I', 'I#', 'II', 'II#', 'III', 'IV', 'IV#', 'V', 'V#', 'VI', 'VI#', 'VII'])
    chromae = np.array(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])

    if shift:
        M = M[shifter][:, shifter]
        chromae = chromae[shifter]

    plt.figure()
    plt.imshow(M)
    plt.xticks(np.arange(12), chromae)
    plt.yticks(np.arange(12), chromae)

    plt.ylim(-.5, 11.5)

    plt.xlabel('From')
    plt.ylabel('To')

    plt.show()



#%%
if __name__ == "__main__":
    if False:
        import numpy as np

        mat = np.array([[20, 6, 1, 11], [5, 18, 8, 7], [6, 5, 20, 6], [16, 4, 5, 12]])
        classnames = ["the beatles", "coldplay", "radiohead", "paul mccartney"]
        imshow_confusion_matrix(mat, classnames)
