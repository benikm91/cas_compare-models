import numpy as np
from sklearn.model_selection import BaseCrossValidator, KFold


class SubsetTrainCV:

    def __init__(self, inner: BaseCrossValidator, subset_percentage: float, random_start_state=None):
        self.inner = inner
        self.subset_percentage = subset_percentage
        if random_start_state is None:
            self.rng = np.random
        else:
            self.rng = np.random.default_rng(random_start_state)
        self.train_sample_sizes = list()

    def _get_mask_for(self, length):
        mask = np.full(length, False)
        mask[:int(length * self.subset_percentage)] = True
        self.rng.shuffle(mask)
        return mask

    def split(self, X, y=None, groups=None):
        for train, test in self.inner.split(X, y, groups):
            train_subset = train[self._get_mask_for(train.shape[0])]
            self.train_sample_sizes.append(len(train_subset))
            yield train_subset, test
