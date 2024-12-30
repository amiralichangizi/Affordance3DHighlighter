import os
import pickle
import torch
import numpy as np

class FullShapeDataset(torch.utils.data.Dataset):
    """
    Loads the full-shape .pkl file and provides coordinate + label data for each object.
    Focuses on common household items with following affordances:
    - grasp: regions suitable for hand grasping
    - push: areas that can be pushed
    - pull: parts that can be pulled
    - lift: regions used for lifting the object
    - move: region for moving
    
    Each entry in the pkl is expected to have:
      - shape_id
      - semantic class
      - affordance (list of strings)
      - full_shape: { 'coordinate': Nx3 array, 'label': { 'grasp': Nx1 array, ... } }
    """
    def __init__(self, pkl_path, device='cuda'):
        super().__init__()
        self.device = device
        
        # Define target household items and affordances
        self.target_classes = [
            'TrashCan', 'Bottle', 'Bowl', 'Bed', 
            'Table', 'Dishwasher', 'Door', 'Chair'
        ]
        
        self.target_affordances = [
            'grasp', 'push', 'pull', 'lift', 'move'
        ]

        # Load and process the .pkl data
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        # Filter data entries for target classes and affordances
        self.data_entries = [
            self._process_entry(entry) for entry in raw_data
            if entry['semantic class'] in self.target_classes and
               any(aff in self.target_affordances for aff in entry['affordance'])
        ]
        if not self.data_entries:
            print("Warning: No valid entries found in the dataset.")

    def _process_entry(self, shape_entry):
        try:
            coords = shape_entry['full_shape']['coordinate']
            labels_dict = shape_entry['full_shape']['label']

            coords_torch = torch.tensor(coords, device=self.device, dtype=torch.float32)
            labels_dict_torch = {
                aff_key: torch.tensor(label_array, device=self.device, dtype=torch.float32).view(-1)
                for aff_key, label_array in labels_dict.items()
                if aff_key in self.target_affordances
            }

            return {
                'shape_id': shape_entry['shape_id'],
                'shape_class': shape_entry['semantic class'],
                'affordances': [aff for aff in shape_entry['affordance'] if aff in self.target_affordances],
                'coords': coords_torch,
                'labels_dict': labels_dict_torch
            }
        except KeyError as e:
            print(f"Missing key in entry {shape_entry.get('shape_id', 'unknown')}: {e}")
            return None

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        return self.data_entries[idx]


from torch.utils.data import Subset

def create_dataset_splits(dataset, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Split the dataset into train, validation, and test subsets.
    Args:
        dataset (FullShapeDataset): The dataset to split.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of test data.
        random_seed (int): Random seed for reproducibility.

    Returns:
        tuple: Subsets for train, validation, and test.
    """
    np.random.seed(random_seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
# returns subset for dataloader.
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)

