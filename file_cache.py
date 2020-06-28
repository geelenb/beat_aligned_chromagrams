import pickle
import os

class FileCache(dict):
    def __init__(self, fname):
        super().__init__()
        self.fname = fname

    def __enter__(self):
        if os.path.exists(self.fname):
            with open(self.fname, 'rb') as f:
                self.update(pickle.load(f))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with open(self.fname, 'wb') as f:
            pickle.dump(self, f)
