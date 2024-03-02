from typing import Optional, Tuple, Sequence
import bisect
import random

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Subset, Sampler, Dataset

from ..datasets.cvat import TrainingPig


class WeightOverSampler(Sampler):
    def __init__(self, dataset: Sequence[TrainingPig], bins: Tuple[int], counts: Tuple[int]):
        """
        Args:
            dataset: dataset instance.
            bins: bins for weight. arbitrary length but has to be same length as counts.
            counts: desired count for each bin.
        """

        super(WeightOverSampler, self).__init__(data_source=None)
        assert len(bins) == len(counts) + 1
        labels = [p.weight for p in dataset]
        _, bins = np.histogram(labels, range=range, bins=bins)
        df = pd.Series(bisect.bisect_left(bins, label) for label in labels)
        weights = pd.Series(counts, index=np.arange(1, len(bins)))
        weights = weights[df]

        self.weights = torch.DoubleTensor(weights.to_list())
        self.length = sum(counts)

    def __len__(self):
        return self.length

    def __iter__(self):
        return map(int, torch.multinomial(self.weights, self.length, replacement=True))


class MinMaxImbalancedSampler(Sampler):
    """
    Oversample given class labels to make sure each class has at least `min_sample_per_individual` samples.
    """
    def __init__(self, labels: np.ndarray, min_sample_per_individual: Optional[int] = 1, 
				max_sample_per_individual: Optional[int] = np.inf):
        super(MinMaxImbalancedSampler, self).__init__(data_source=None)
        self.labels = labels
        self.label_to_count = pd.Series(labels).value_counts()
        self.min_sample_per_individual = min_sample_per_individual
        self.max_sample_per_individual = max_sample_per_individual

    def __len__(self):
        return np.clip(
				np.array(self.label_to_count), 
				a_min=self.min_sample_per_individual, 
				a_max=self.max_sample_per_individual
			).sum()

    def __iter__(self):
        indices = []
        for label, count in self.label_to_count.iteritems():
            _indices = np.where(self.labels == label)[0]
            if count > self.max_sample_per_individual:
                indices.extend(np.random.choice(_indices, self.max_sample_per_individual, replace=False).tolist())
            elif count < self.min_sample_per_individual:
                indices.extend(np.random.choice(_indices, self.min_sample_per_individual, replace=True).tolist())
            else:
                indices.extend(_indices.tolist())
        random.shuffle(indices)
        return iter(indices)


class IndividualImbalancedSampler(MinMaxImbalancedSampler):
    def __init__(self, dataset: Sequence[TrainingPig], min_sample_per_individual: int, max_sample_per_individual: int):
        """
        Args:
            dataset: dataset class
            min_sample_per_individual: minimum number of samples per individual.
            max_sample_per_individual: maximum number of samples per individual.
        """
        ids = np.array([pig.id + str(pig.weight) for pig in dataset])
        super(IndividualImbalancedSampler, self).__init__(
			abels=ids, 
			min_sample_per_individual=min_sample_per_individual, 
			max_sample_per_individual=max_sample_per_individual)


def get_oversampled_dataset(dataset: Dataset, sampler: Sampler):
    return Subset(dataset=dataset, indices=list(sampler))
