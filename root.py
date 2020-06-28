import pickle

import numpy as np

np.warnings.filterwarnings("ignore")



from collections import Counter

with open('gtzan.p', 'rb') as f:
    dataset = pickle.load(f)

sum([
    1
    for pitches in dataset.list_of_pitch_matrices
    if Counter(np.argmax(pitches[:, :12], 1)).most_common(1)[0][0] == 0
])
