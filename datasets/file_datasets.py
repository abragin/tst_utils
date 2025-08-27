from datasets import Dataset
import pandas as pd
from torch.utils.data import IterableDataset

class SingleFileTinyStylerDataset(IterableDataset):
    """
    Wraps a single parquet file into an infinite or finite IterableDataset.
    Uses HuggingFace datasets for shuffle and processing.
    """

    def __init__(self, file_path, ds_processing_fn, infinite=True, shuffle_seed=None):
        super().__init__()
        self.file_path = file_path
        self.ds_processing_fn = ds_processing_fn
        self.infinite = infinite
        self.shuffle_seed = shuffle_seed

        # Load & prepare once
        df = pd.read_parquet(self.file_path)
        ds = Dataset.from_pandas(df)
        ds = self.ds_processing_fn(ds)
        self.ds = ds

    def __iter__(self):
        epoch = 0
        while True:
            # reshuffle each epoch
            shuffled = self.ds.shuffle(seed=(self.shuffle_seed + epoch) if self.shuffle_seed else None)
            for row in shuffled:
                yield row

            if not self.infinite:
                break
            epoch += 1


class MultiFileTinyStylerDataset(IterableDataset):
    """
    Cycles through multiple parquet files in sequence.
    Each file is loaded into SingleFileTinyStylerDataset internally.
    """

    def __init__(self, file_paths, ds_processing_fn, infinite=True, shuffle_seed=None):
        super().__init__()
        self.file_paths = file_paths
        self.ds_processing_fn = ds_processing_fn
        self.infinite = infinite
        self.shuffle_seed = shuffle_seed

    def __iter__(self):
        file_idx = 0
        while True:
            ds = SingleFileTinyStylerDataset(
                self.file_paths[file_idx],
                self.ds_processing_fn,
                infinite=False,  # one pass per file
                shuffle_seed=self.shuffle_seed,
            )
            for row in ds:
                yield row

            file_idx = (file_idx + 1) % len(self.file_paths)
            if not self.infinite and file_idx == 0:
                break