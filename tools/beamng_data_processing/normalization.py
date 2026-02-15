"""Data normalization utilities for BeamNG vehicle dynamics."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


class DataNormalizer:
    """Z-score normalization for inputs and targets."""

    def __init__(
        self,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ):
        """
        Args:
            input_mean: Mean of input features (excluding time column)
            input_std: Std of input features (excluding time column)
            target_mean: Mean of target features (excluding time column)
            target_std: Std of target features (excluding time column)
        """
        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std

        # Avoid division by zero
        self.input_std = np.where(input_std < 1e-8, 1.0, input_std)
        self.target_std = np.where(target_std < 1e-8, 1.0, target_std)

    def normalize_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Normalize inputs (z-score).

        Args:
            inputs: [batch, n_points, n_features] or [n_points, n_features]
                First column is time (not normalized), rest are features

        Returns:
            Normalized inputs with same shape
        """
        # Convert to numpy for computation
        is_tensor = isinstance(inputs, torch.Tensor)
        device = inputs.device if is_tensor else None
        inputs_np = inputs.cpu().numpy() if is_tensor else inputs

        # Separate time and features
        time = inputs_np[..., 0:1]  # Keep time as-is
        features = inputs_np[..., 1:]  # Normalize features

        # Z-score normalization
        features_norm = (features - self.input_mean) / self.input_std

        # Concatenate back
        result = np.concatenate([time, features_norm], axis=-1)

        # Convert back to tensor if needed
        if is_tensor:
            result = torch.from_numpy(result).float().to(device)

        return result

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Normalize targets (z-score).

        Args:
            targets: [batch, n_points, n_features] or [n_points, n_features]
                First column is time (not normalized), rest are Δ values

        Returns:
            Normalized targets with same shape
        """
        # Convert to numpy for computation
        is_tensor = isinstance(targets, torch.Tensor)
        device = targets.device if is_tensor else None
        targets_np = targets.cpu().numpy() if is_tensor else targets

        # Separate time and features
        time = targets_np[..., 0:1]  # Keep time as-is
        features = targets_np[..., 1:]  # Normalize Δ values

        # Z-score normalization
        features_norm = (features - self.target_mean) / self.target_std

        # Concatenate back
        result = np.concatenate([time, features_norm], axis=-1)

        # Convert back to tensor if needed
        if is_tensor:
            result = torch.from_numpy(result).float().to(device)

        return result

    def denormalize_targets(self, targets_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize targets back to original scale.

        Args:
            targets_norm: Normalized targets [batch, n_points, n_features]

        Returns:
            Denormalized targets with same shape
        """
        # Convert to numpy for computation
        is_tensor = isinstance(targets_norm, torch.Tensor)
        device = targets_norm.device if is_tensor else None
        targets_np = targets_norm.cpu().numpy() if is_tensor else targets_norm

        # Separate time and features
        time = targets_np[..., 0:1]
        features_norm = targets_np[..., 1:]

        # Inverse z-score
        features = features_norm * self.target_std + self.target_mean

        # Concatenate back
        result = np.concatenate([time, features], axis=-1)

        # Convert back to tensor if needed
        if is_tensor:
            result = torch.from_numpy(result).float().to(device)

        return result

    def save(self, path: Path) -> None:
        """Save normalization parameters to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DataNormalizer":
        """Load normalization parameters from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            input_mean=np.array(data["input_mean"]),
            input_std=np.array(data["input_std"]),
            target_mean=np.array(data["target_mean"]),
            target_std=np.array(data["target_std"]),
        )


def compute_normalization_params(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
) -> DataNormalizer:
    """
    Compute normalization parameters from training data.

    Args:
        train_inputs: [n_samples, n_features] - first column is time
        train_targets: [n_samples, n_features] - first column is time

    Returns:
        DataNormalizer with computed statistics
    """
    # Compute statistics (excluding time column)
    input_features = train_inputs[:, 1:]  # Skip time
    target_features = train_targets[:, 1:]  # Skip time

    input_mean = input_features.mean(axis=0)
    input_std = input_features.std(axis=0)
    target_mean = target_features.mean(axis=0)
    target_std = target_features.std(axis=0)

    return DataNormalizer(input_mean, input_std, target_mean, target_std)


def print_normalization_stats(normalizer: DataNormalizer) -> None:
    """Print normalization statistics."""
    def _expand_names(names, count, extra_name=None, insert_at=None):
        if len(names) == count:
            return names
        if extra_name and len(names) + 1 == count:
            names = list(names)
            if insert_at is None:
                names.append(extra_name)
            else:
                names.insert(insert_at, extra_name)
            return names
        names = list(names)
        if len(names) < count:
            names.extend([f"feat_{i}" for i in range(len(names), count)])
        return names[:count]

    input_names = ["x", "y", "yaw", "vx", "vy", "vyaw", "brake", "parking_brake", "steer", "throttle"]
    target_names = ["Δx", "Δy", "Δyaw", "Δvx", "Δvy", "Δvyaw"]
    input_names = _expand_names(input_names, len(normalizer.input_mean), extra_name="engine_rpm", insert_at=6)
    target_names = _expand_names(target_names, len(normalizer.target_mean), extra_name="Δrpm")

    print("\n=== Normalization Parameters ===")
    print("\nInput Features:")
    print(f"{'Feature':<15s} {'Mean':<12s} {'Std':<12s}")
    print("-" * 40)
    for name, mean, std in zip(input_names, normalizer.input_mean, normalizer.input_std):
        print(f"{name:<15s} {mean:>12.6f} {std:>12.6f}")

    print("\nTarget Features:")
    print(f"{'Feature':<15s} {'Mean':<12s} {'Std':<12s}")
    print("-" * 40)
    for name, mean, std in zip(target_names, normalizer.target_mean, normalizer.target_std):
        print(f"{name:<15s} {mean:>12.6f} {std:>12.6f}")

    print("\n")
