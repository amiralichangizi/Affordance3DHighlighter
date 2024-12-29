
import os
import pickle
import torch
import numpy as np

class FullShapeDataset(torch.utils.data.Dataset):
    """
    Loads the full-shape .pkl file and provides coordinate + label data for each object.
    Each entry in the pkl is expected to have:
      - shape_id
      - semantic class
      - affordance (list of strings)
      - full_shape: { 'coordinate': Nx3 array, 'label': { 'grasp': Nx1 array, ... } }
    """
    def __init__(self, pkl_path, device='cuda'):
        super().__init__()
        self.device = device

        # Load the pkl data
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)

        # raw_data is typically a list of shape entries
        self.data_entries = []
        for shape_entry in raw_data:
            shape_id = shape_entry['shape_id']
            shape_class = shape_entry['semantic class']
            affordances = shape_entry['affordance']  # e.g. ['grasp', 'pushable']
            
            # Full shape info
            coords = shape_entry['full_shape']['coordinate']  # Nx3
            labels_dict = shape_entry['full_shape']['label']  # {'grasp': Nx1 array, etc.}

            # Convert to Torch
            coords_torch = torch.tensor(coords, device=self.device, dtype=torch.float32)

            # Convert each label to Torch (still Nx1 or Nx?)
            labels_dict_torch = {}
            for aff_key, label_array in labels_dict.items():
                # e.g. shape Nx1 or Nx
                label_torch = torch.tensor(label_array, device=self.device, dtype=torch.float32).squeeze()
                # shape [N], each entry is 0 or 1 (?)
                labels_dict_torch[aff_key] = label_torch

            entry = {
                'shape_id': shape_id,
                'shape_class': shape_class,
                'affordances': affordances,
                'coords': coords_torch,          # [N, 3]
                'labels_dict': labels_dict_torch # {aff_type: [N], ...}
            }
            self.data_entries.append(entry)

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        return self.data_entries[idx]
