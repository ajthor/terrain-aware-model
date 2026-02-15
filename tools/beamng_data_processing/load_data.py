import os

from typing import List, Optional

import csv

import numpy as np

import torch
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path

# Import normalization
try:
    from .normalization import DataNormalizer
except ImportError:
    from normalization import DataNormalizer



class TrainDataset(IterableDataset):
    def __init__(
        self,
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor],
        n_example_points: int,
        n_points: int,
        normalizer: Optional[DataNormalizer] = None,
    ):

        self.inputs = inputs  # list of tensors of shape [n_points, n_features]
        self.targets = targets  # list of tensors of shape [n_points, n_features]
        self.normalizer = normalizer

        assert len(self.inputs) == len(
            self.targets
        ), "Inputs and targets must have the same length"

        for i in range(len(self.inputs)):
            if self.inputs[i].shape[0] != self.targets[i].shape[0]:
                raise ValueError(
                    f"Input and target tensors must have the same number of samples. "
                    f"Input shape: {self.inputs[i].shape}, Target shape: {self.targets[i].shape}"
                )

        self.n_example_points = n_example_points
        self.n_points = n_points
        self.xs_mean = None
        self.xs_std = None
        self.ys_mean = None
        self.ys_std = None

    def __iter__(self):
        while True:
            n_samples = self.n_points + self.n_example_points

            # Generate a random index to select a scene
            B = torch.randint(0, len(self.inputs), (1,)).item()

            inputs = self.inputs[B]
            targets = self.targets[B]

            # Random sampling of points from the training data
            indices = torch.randperm(inputs.shape[0])[:n_samples]

            # Get inputs and targets
            sampled_inputs = inputs[indices]  # [n_samples, n_features]
            sampled_targets = targets[indices]  # [n_samples, n_features]

            # Apply normalization if provided
            if self.normalizer is not None:
                sampled_inputs = self.normalizer.normalize_inputs(sampled_inputs)
                sampled_targets = self.normalizer.normalize_targets(sampled_targets)

            _xs = sampled_inputs[:, 1:]  # Drop time column
            _dt = sampled_targets[:, 0] - sampled_inputs[:, 0]  # Time difference
            _ys = sampled_targets[:, 1:]  # Drop time column

            # Split the data
            example_xs = _xs[: self.n_example_points]
            example_dt = _dt[: self.n_example_points]
            example_ys = _ys[: self.n_example_points]

            xs = _xs[self.n_example_points :]
            dt = _dt[self.n_example_points :]
            ys = _ys[self.n_example_points :]

            yield xs, dt, ys, example_xs, example_dt, example_ys


class TestDataset(Dataset):
    def __init__(
        self,
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor],
        n_example_points: int,
        normalizer: Optional[DataNormalizer] = None,
    ):
        self.inputs = inputs
        self.targets = targets
        self.n_example_points = n_example_points
        self.normalizer = normalizer

        self.xs_mean = None
        self.xs_std = None
        self.ys_mean = None
        self.ys_std = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        # Sample random points from the data without replacement
        indices = torch.randperm(self.inputs[idx].shape[0])
        example_indices = indices[: self.n_example_points]
        query_indices = indices[self.n_example_points :]

        # Get inputs and targets
        inputs = self.inputs[idx]
        targets = self.targets[idx]

        # Apply normalization if provided
        if self.normalizer is not None:
            inputs = self.normalizer.normalize_inputs(inputs)
            targets = self.normalizer.normalize_targets(targets)

        _xs = inputs[:, 1:]
        _dt = targets[:, 0] - inputs[:, 0]
        _ys = targets[:, 1:]
        example_xs = _xs[example_indices]
        example_dt = _dt[example_indices]
        example_ys = _ys[example_indices]
        xs = _xs[query_indices]
        dt = _dt[query_indices]
        ys = _ys[query_indices]

        return xs, dt, ys, example_xs, example_dt, example_ys


class AllDataset(Dataset):
    def __init__(
        self,
        inputs: List[torch.Tensor],
        targets: List[torch.Tensor],
        n_example_points: int,
    ):
        self.inputs = inputs
        self.targets = targets
        self.n_example_points = n_example_points

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        # Sample random points from the data without replacement
        indices = torch.randperm(self.inputs[idx].shape[0])
        example_indices = indices[: self.n_example_points]

        xs = self.inputs[idx][:, 1:]
        dt = self.targets[idx][:, 0] - self.inputs[idx][:, 0]
        ys = self.targets[idx][:, 1:]
        example_xs = xs[example_indices]
        example_dt = dt[example_indices]
        example_ys = ys[example_indices]

        return xs, dt, ys, example_xs, example_dt, example_ys


def load_csv(filepath):
    """
    Load a CSV file and return the data as a tensor.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))  # path to terrain_adaptation_rls_rls/data
    repo_root = os.path.abspath(os.path.join(repo_root, "../.."))  # path to repo root
    full_path = os.path.join(repo_root, filepath)
    data = []
    with open(full_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)