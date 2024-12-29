
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

        # Load the pkl data
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)

        # Filter and process the data
        self.data_entries = []
        for shape_entry in raw_data:
            shape_id = shape_entry['shape_id']
            shape_class = shape_entry['semantic class']
            
            # Only include target household items
            if shape_class not in self.target_classes:
                continue
                
            affordances = [aff for aff in shape_entry['affordance'] 
                         if aff in self.target_affordances]
            
            if not affordances:  # Skip if no target affordances
                continue
            
            # Full shape info
            coords = shape_entry['full_shape']['coordinate']
            labels_dict = shape_entry['full_shape']['label']

            # Convert to Torch
            coords_torch = torch.tensor(coords, device=self.device, dtype=torch.float32)

            # Convert each label to Torch
            labels_dict_torch = {}
            for aff_key in affordances:
                label_array = labels_dict[aff_key]
                label_torch = torch.tensor(label_array, device=self.device, 
                                         dtype=torch.float32).squeeze()
                labels_dict_torch[aff_key] = label_torch

            entry = {
                'shape_id': shape_id,
                'shape_class': shape_class,
                'affordances': affordances,
                'coords': coords_torch,
                'labels_dict': labels_dict_torch
            }
            self.data_entries.append(entry)

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        return self.data_entries[idx]
