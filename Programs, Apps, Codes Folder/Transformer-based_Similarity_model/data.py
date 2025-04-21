# data.py
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time # Optional: for timing pair generation

# 自定义数据集类 (No changes needed here, keep as is)
class SkeletonPairDataset(Dataset):
    def __init__(self, pairs, labels, max_frames=100):
        self.pairs = pairs
        self.labels = labels
        self.max_frames = max_frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path1, path2 = self.pairs[idx]
        tensor1 = self.load_and_process(str(path1))
        tensor2 = self.load_and_process(str(path2))
        label = self.labels[idx]
        # Ensure label is returned as a tensor for consistency if needed later,
        # but DataLoader usually handles batching scalars correctly.
        # return tensor1, tensor2, torch.tensor(label, dtype=torch.int64)
        return tensor1, tensor2, label # Return label as is (int)

    def load_and_process(self, file_path):
        """加载并处理单个骨架文件"""
        try:
            # Load data using np.loadtxt
            data = np.loadtxt(file_path)

            # Handle potential 1D array (single frame or single feature across frames)
            if data.ndim == 0: # Handle scalar case (e.g., empty file loaded as 0)
                 print(f"Warning: File {file_path} loaded as scalar. Returning zeros.")
                 return torch.zeros(self.max_frames, 63, dtype=torch.float32)
            elif data.ndim == 1:
                # Assumption: If 1D, it's likely a single frame with 64 features. Reshape.
                if data.shape[0] == 64:
                     data = np.expand_dims(data, axis=0) # Shape becomes (1, 64)
                # Add handling for other 1D shapes if necessary, otherwise it might fail below
                else:
                     print(f"Warning: File {file_path} loaded as 1D array with unexpected shape {data.shape}. Returning zeros.")
                     return torch.zeros(self.max_frames, 63, dtype=torch.float32)


            # --- Check number of columns BEFORE processing ---
            if data.shape[1] != 64:
                print(f"Error: File {file_path} has {data.shape[1]} columns, expected 64. Cannot process reliably.")
                # Returning zeros as per previous logic for consistency
                return torch.zeros(self.max_frames, 63, dtype=torch.float32)

            # Convert to tensor
            tensor = torch.tensor(data, dtype=torch.float32)

            # --- Skip the first column (feature) ---
            # Assuming the first column is always an index/timestamp to be removed
            tensor = tensor[:, 1:] # Shape becomes (current_frames, 63)

            # Pad or truncate frames (dim=0)
            current_frames = tensor.shape[0]
            num_features = tensor.shape[1] # Should be 63 now

            if current_frames == 0: # Check if tensor became empty after column skip
                 print(f"Warning: Tensor became empty after skipping column for file {file_path}. Returning zeros.")
                 return torch.zeros(self.max_frames, 63, dtype=torch.float32)


            if current_frames < self.max_frames:
                # Pad frames
                pad_size = self.max_frames - current_frames
                pad = torch.zeros(pad_size, num_features)
                tensor = torch.cat([tensor, pad], dim=0)
            elif current_frames > self.max_frames:
                # Truncate frames
                tensor = tensor[:self.max_frames, :]

            # Final shape check
            if tensor.shape != (self.max_frames, 63):
                 print(f"Warning: Final tensor shape is {tensor.shape} for file {file_path}, expected ({self.max_frames}, 63).")
                 # This case ideally shouldn't be reached with the checks above
                 # If it does, consider padding/truncating features if necessary, but it indicates an issue.
                 # Example: Pad features if fewer than 63
                 if tensor.shape[1] < 63:
                     pad_feat = torch.zeros(self.max_frames, 63 - tensor.shape[1])
                     tensor = torch.cat([tensor, pad_feat], dim=1)
                 # Example: Truncate features if more than 63 (unlikely)
                 elif tensor.shape[1] > 63:
                     tensor = tensor[:, :63]


            return tensor

        except FileNotFoundError:
             print(f"Error: File not found during loading: {file_path}")
             return torch.zeros(self.max_frames, 63, dtype=torch.float32)
        except ValueError as ve: # Catches errors during np.loadtxt if format is wrong
             print(f"Error loading file {file_path} with numpy.loadtxt (check format): {ve}")
             return torch.zeros(self.max_frames, 63, dtype=torch.float32)
        except Exception as e:
            print(f"Unexpected error loading or processing file {file_path}: {e}")
            # Return a dummy tensor
            return torch.zeros(self.max_frames, 63, dtype=torch.float32)


# --- Modified build_dataloaders for the NEW dataset structure ---
def build_dataloaders(data_dir, max_frames=100, batch_size=32, test_split=0.2, random_state=42, num_workers=2, target_pairs_per_class=1000):
    """
    Builds training and testing dataloaders from the RESTRUCTURED dataset.

    Args:
        data_dir (str): Path to the restructured data directory (contains action folders).
        max_frames (int): Maximum frames for padding/truncation.
        batch_size (int): Batch size for DataLoaders.
        test_split (float): Proportion of data for the test set.
        random_state (int): Random seed for reproducible splits.
        num_workers (int): Number of worker processes for DataLoader.
        target_pairs_per_class (int): Desired number of positive/negative pairs.

    Returns:
        tuple: (train_loader, test_loader)
    """
    print(f"Scanning NEW data structure in directory: {data_dir}")
    action_samples = {} # {action_name: [list_of_skeleton_paths]}
    start_scan_time = time.time()

    # --- Updated Loop Structure for New Dataset ---
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Iterate through items in the base data directory (expecting action folders)
    for action_name in os.listdir(data_dir):
        action_path = os.path.join(data_dir, action_name)
        if not os.path.isdir(action_path): # Skip files, only process action directories
            # print(f"Skipping non-directory item: {action_name}")
            continue

        # Valid action directory found
        action_samples[action_name] = [] # Initialize list for this action

        # Iterate through items in the action directory (expecting instance folders)
        for instance_name in os.listdir(action_path):
            instance_path = os.path.join(action_path, instance_name)
            if not os.path.isdir(instance_path): # Skip files, only process instance directories
                # print(f"Skipping non-directory item in {action_name}: {instance_name}")
                continue

            # Valid instance directory found
            skeleton_path = os.path.join(instance_path, 'skeleton.txt')

            if os.path.exists(skeleton_path):
                # Check if non-empty (restructuring should ensure this, but double-check)
                if os.path.getsize(skeleton_path) > 0:
                    action_samples[action_name].append(skeleton_path)
                else:
                    print(f"Warning: Found empty skeleton file (should have been removed by restructuring): {skeleton_path}")
            # else: No skeleton.txt found in this instance folder, skip

        # Remove action from dict if no valid skeletons were found within it
        if not action_samples[action_name]:
            del action_samples[action_name]
            print(f"Warning: No valid skeleton files found for action: {action_name}")
    # --- End of Updated Loop Structure ---

    scan_duration = time.time() - start_scan_time
    print(f"Directory scan completed in {scan_duration:.2f} seconds.")

    if not action_samples:
        raise ValueError(f"No valid skeleton files found in the specified directory structure: {data_dir}")
    print(f"Found {len(action_samples)} actions with valid skeleton data.")
    for name, samples in action_samples.items():
        print(f"  - Action '{name}': {len(samples)} samples")

    # --- Pair Generation Logic (largely unchanged, uses populated action_samples) ---
    print("\nGenerating pairs...")
    start_pair_time = time.time()

    valid_actions_for_pos = [act for act, samps in action_samples.items() if len(samps) >= 2]
    valid_actions_for_neg = list(action_samples.keys()) # Any action can be used for negative pairs

    if len(valid_actions_for_pos) < 1:
        raise ValueError("Not enough actions with at least 2 samples to create positive pairs.")
    if len(valid_actions_for_neg) < 2:
        raise ValueError("Need at least 2 distinct actions with samples to create negative pairs.")

    positive_pairs = []
    negative_pairs = []
    max_attempts_multiplier = 20 # Increase multiplier for potentially harder pair finding

    print(f"Targeting {target_pairs_per_class} positive pairs...")
    attempts = 0
    max_attempts = target_pairs_per_class * max_attempts_multiplier
    positive_pairs_set = set() # Use a set for faster duplicate checking

    while len(positive_pairs) < target_pairs_per_class and attempts < max_attempts:
        action = random.choice(valid_actions_for_pos)
        samples = action_samples[action]
        try:
            s1, s2 = random.sample(samples, 2)
            pair_tuple = tuple(sorted((s1, s2))) # Canonical representation for the pair
            if pair_tuple not in positive_pairs_set:
                 positive_pairs.append([s1, s2]) # Append original order list
                 positive_pairs_set.add(pair_tuple)
        except ValueError:
             pass # Should not happen often due to valid_actions_for_pos check
        attempts += 1
        if attempts % 1000 == 0: # Progress indicator
             print(f"  Positive pair attempts: {attempts}, Found: {len(positive_pairs)}")

    if len(positive_pairs) < target_pairs_per_class:
        print(f"Warning: Could only generate {len(positive_pairs)} unique positive pairs after {attempts} attempts.")

    print(f"\nTargeting {target_pairs_per_class} negative pairs...")
    attempts = 0
    max_attempts = target_pairs_per_class * max_attempts_multiplier
    negative_pairs_set = set() # Use a set for faster duplicate checking

    while len(negative_pairs) < target_pairs_per_class and attempts < max_attempts:
        try:
            a1, a2 = random.sample(valid_actions_for_neg, 2) # Sample two distinct action names
            s1 = random.choice(action_samples[a1])
            s2 = random.choice(action_samples[a2])
            pair_tuple = tuple(sorted((s1, s2))) # Canonical representation

            # Ensure it's not accidentally a positive pair or a duplicate negative
            if pair_tuple not in positive_pairs_set and pair_tuple not in negative_pairs_set:
                negative_pairs.append([s1, s2]) # Append original order list
                negative_pairs_set.add(pair_tuple)
        except ValueError: # Should not happen if len(valid_actions_for_neg) >= 2
             break
        except IndexError: # If random.choice fails on an empty list (shouldn't happen)
             pass
        attempts += 1
        if attempts % 1000 == 0: # Progress indicator
             print(f"  Negative pair attempts: {attempts}, Found: {len(negative_pairs)}")


    if len(negative_pairs) < target_pairs_per_class:
        print(f"Warning: Could only generate {len(negative_pairs)} unique negative pairs after {attempts} attempts.")

    pair_duration = time.time() - start_pair_time
    print(f"Pair generation completed in {pair_duration:.2f} seconds.")

    # --- Combine, Split, Create Datasets and DataLoaders (unchanged) ---
    all_pairs = positive_pairs + negative_pairs
    all_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    if not all_pairs:
        raise ValueError("No pairs were generated. Check data and pairing logic.")
    print(f"\nTotal pairs generated: {len(all_pairs)} (Positive: {len(positive_pairs)}, Negative: {len(negative_pairs)})")

    print(f"Splitting data: {1-test_split:.0%} train, {test_split:.0%} test...")
    if len(all_pairs) < 2:
         raise ValueError(f"Only {len(all_pairs)} pair generated. Cannot split into train/test sets.")
    if len(np.unique(all_labels)) < 2 and len(all_pairs) > 0:
         print("Warning: Only one class present in the generated pairs. Stratified split might behave unexpectedly or fail.")
         # Use non-stratified split as fallback
         train_pairs, test_pairs, train_labels, test_labels = train_test_split(
            all_pairs, all_labels, test_size=test_split, random_state=random_state, shuffle=True
        )
    elif len(all_pairs) > 0 :
         train_pairs, test_pairs, train_labels, test_labels = train_test_split(
             all_pairs, all_labels, test_size=test_split, random_state=random_state, stratify=all_labels
        )
    else: # Should be caught by "No pairs generated" check, but belts and suspenders
         train_pairs, test_pairs, train_labels, test_labels = [], [], [], []


    print(f"Train samples: {len(train_pairs)}, Test samples: {len(test_pairs)}")
    if len(train_pairs) == 0 or len(test_pairs) == 0:
        print("Warning: Train or Test set is empty after split. Check pair generation and split ratio.")


    print("Creating Datasets and DataLoaders...")
    train_dataset = SkeletonPairDataset(train_pairs, train_labels, max_frames)
    test_dataset = SkeletonPairDataset(test_pairs, test_labels, max_frames)

    # Use the num_workers argument passed to the function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0)

    print("DataLoaders created successfully.")
    return train_loader, test_loader
