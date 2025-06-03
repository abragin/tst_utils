import numpy as np
import torch

class MixedDataset(torch.utils.data.IterableDataset):
    def __init__(self, datasets, weights):
        assert len(datasets) == len(weights), "Each dataset must have a matching weight."
        w_sum = sum(weights)
        probabilities = [w/w_sum for w in weights]
        self.datasets = datasets
        self.probabilities = probabilities
        self.iterators = [iter(ds) for ds in datasets]

    def __iter__(self):
        self.iterators = [iter(ds) for ds in self.datasets]  # reset iterators
        while True:
            i = np.random.choice(len(self.datasets), p=self.probabilities)
            try:
                yield next(self.iterators[i])
            except StopIteration:
                # Reinitialize exhausted iterator
                self.iterators[i] = iter(self.datasets[i])
                yield next(self.iterators[i])