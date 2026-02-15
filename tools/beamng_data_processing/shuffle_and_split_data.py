import argparse
import csv
import os
import numpy as np
import pandas as pd
import torch

from process_data import process_data

def write_csv(path, tensor_data):
    """Write a torch tensor to a CSV file."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in tensor_data.tolist():
            writer.writerow(row)


def check_time_alignment(csv1, csv2):
    pd1 = pd.read_csv(csv1)
    pd2 = pd.read_csv(csv2)
    t1 = pd1['time'].to_numpy()
    t2 = pd2['time'].to_numpy()
    n = min(len(t1), len(t2))
    if n == 0:
        print("⚠️ Time Diff = nan +/- nan (empty time series)")
        return
    if len(t1) != len(t2):
        print(f"⚠️ Time series length mismatch: {len(t1)} vs {len(t2)} (using first {n})")
    del_t = t1[:n] - t2[:n]

    if np.abs(np.mean(del_t)) < 1e-3:
        print(f"✅ Time Diff = {np.mean(del_t):.1e} +/- {np.std(del_t):.1e}")
    else:
        print(f"⚠️ Time Diff = {np.mean(del_t):.1e} +/- {np.std(del_t):.1e}")




def main():

    print("\n============== SHUFFLING AND SPLITTING DATA ===============\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", type=str, default='asphalt')
    parser.add_argument("--transmission", type=str, default='m1')
    parser.add_argument("--type", type=str, default='train')
    parser.add_argument("--chronological", action='store_true', default=True,
                        help='Use chronological split: first 80%% train, last 20%% test. '
                             'Most realistic for real-world deployment. (DEFAULT)')
    parser.add_argument("--random-shuffle", action='store_true',
                        help='Use random point shuffling instead of chronological split. '
                             'Only for ablation studies.')
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help='Fraction of data to use for training (default: 0.8).')
    parser.add_argument("--include-rpm", action="store_true",
                        help="Include engine RPM as a state (adds Δrpm target).")
    args = parser.parse_args()

    # Folder containing all terrain subfolders
    base_dir = f"function_encoder_beamng/data/{args.transmission}/{args.type}/{args.terrain}"

    # Save CSV data to numpy arrays
    state_csv = os.path.join(base_dir, f"state_sensor.csv")
    angvel_csv = os.path.join(base_dir, f"imu_angular_velocity.csv")
    electrics_csv = os.path.join(base_dir, f"controls.csv")

    # Check time alignment between data files
    check_time_alignment(state_csv, angvel_csv)
    check_time_alignment(state_csv, electrics_csv)
    check_time_alignment(angvel_csv, electrics_csv)

    # Read CSV as numpy array, skipping header
    states = pd.read_csv(state_csv, header=None, skiprows=1).to_numpy()
    angvels = pd.read_csv(angvel_csv, header=None, skiprows=1).to_numpy()
    electrics = pd.read_csv(electrics_csv)

    # Process the data into inputs and targets.
    if args.include_rpm:
        print("Including engine RPM as state (Δrpm target).")
    inputs, targets = process_data(states, angvels, electrics, include_rpm=args.include_rpm)
    print("\nInputs: ", inputs.shape)
    print("Targets: ", targets.shape)

    # # z-Normalize the target data (NOT TIME!). 
    # mean = targets[:,1:].mean(dim=0, keepdim=True)
    # std = targets[:,1:].std(dim=0, keepdim=True)

    # # Avoid divide-by-zero
    # std[std == 0] = 1.0

    # # z-normalize
    # targets[:,1:] = (targets[:,1:] - mean) / std


    # Split data for different seeds.
    seeds = [0, 1, 2, 3, 4]
    total_points = inputs.shape[0]

    for seed in seeds:

        # Set the random seed.
        torch.manual_seed(seed)
        np.random.seed(seed)

        if args.chronological and not args.random_shuffle:
            # CHRONOLOGICAL SPLIT: First X% train, last (1-X)% test
            # This is the most realistic for real-world deployment
            # Note: seed doesn't affect the split since no shuffling occurs,
            # but we still save to seed folders for consistency with training scripts
            split_idx = int(args.train_ratio * total_points)
            
            train_input = inputs[:split_idx]
            train_target = targets[:split_idx]
            test_input = inputs[split_idx:]
            test_target = targets[split_idx:]
            
            # Calculate time spans for reporting
            train_time_start = train_input[0, 0].item()
            train_time_end = train_input[-1, 0].item()
            test_time_start = test_input[0, 0].item()
            test_time_end = test_input[-1, 0].item()
            
            print(f"\n[Seed {seed}] Chronological split ({args.train_ratio*100:.0f}% train / {(1-args.train_ratio)*100:.0f}% test)")
            print(f"  Train: {len(train_input)} points (t={train_time_start:.1f}s to {train_time_end:.1f}s)")
            print(f"  Test:  {len(test_input)} points (t={test_time_start:.1f}s to {test_time_end:.1f}s)")

        else:
            # RANDOM POINT SHUFFLING: For ablation studies only
            # NOT recommended for real deployment - causes temporal leakage
            print(f"\n[Seed {seed}] Random point shuffling (WARNING: temporal leakage, use --chronological instead)")
            indices = torch.randperm(total_points)

            # Get the split indices for 80% training and 20% testing.
            split_idx = int(0.8 * total_points)
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]

            # Split the data into train and test sets.
            train_input = inputs[train_indices]
            train_target = targets[train_indices]
            test_input = inputs[test_indices]
            test_target = targets[test_indices]

            print(f"[Seed {seed}] Train: {len(train_indices)} points, Test: {len(test_indices)} points")

        # Compute and save normalization parameters from training data
        from normalization import compute_normalization_params, print_normalization_stats

        normalizer = compute_normalization_params(
            train_input.numpy(),
            train_target.numpy(),
        )

        print(f"\n[Seed {seed}] Computing normalization parameters...")
        print_normalization_stats(normalizer)

        # Save the data to CSV files
        save_path = f"function_encoder_beamng/data_split/seed_{seed}/{args.transmission}/{args.terrain}"
        os.makedirs(save_path, exist_ok=True)
        write_csv(f"{save_path}/train_input.csv", train_input)
        write_csv(f"{save_path}/train_target.csv", train_target)
        write_csv(f"{save_path}/test_input.csv", test_input)
        write_csv(f"{save_path}/test_target.csv", test_target)

        # Save normalization parameters
        from pathlib import Path
        norm_path = Path(save_path) / "normalization.json"
        normalizer.save(norm_path)
        print(f"[Seed {seed}] Saved normalization parameters to {norm_path}")


if __name__ == '__main__':
    main()
