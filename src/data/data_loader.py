# src/data/data_loader.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class ReflexDataset(Dataset):
    """Dataset class for RefleX X-ray diffraction images."""
    
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Read labels file with explicit type conversion
        self.labels_df = pd.read_csv(labels_file)
        
        # Define the label columns
        self.label_columns = [
            'loop_scattering',
            'background_ring',
            'strong_background',
            'diffuse_scattering',
            'artifact',
            'ice_ring',
            'non_uniform_detector'
        ]
        
        # Convert label columns to float32
        for col in self.label_columns:
            self.labels_df[col] = pd.to_numeric(self.labels_df[col], errors='coerce').fillna(0).astype(np.float32)
        
        # Verify data loading
        print(f"Loaded dataset with {len(self.labels_df)} samples")
        print("Label columns dtype check:")
        for col in self.label_columns:
            print(f"{col}: {self.labels_df[col].dtype}")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        """Returns a sample from the dataset."""
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
            
        # Get image path and load image
        img_name = str(self.labels_df.iloc[idx]['image'])
        if not img_name.endswith('.png'):
            img_name = f"{img_name}.png"
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            raise
            
        # Get labels with explicit type conversion
        try:
            labels = self.labels_df.iloc[idx][self.label_columns].values.astype(np.float32)
            labels = torch.from_numpy(labels)
        except Exception as e:
            print(f"Error getting labels for index {idx}: {str(e)}")
            print(f"Values: {self.labels_df.iloc[idx][self.label_columns].values}")
            print(f"Types: {[type(x) for x in self.labels_df.iloc[idx][self.label_columns].values]}")
            raise
            
        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'labels': labels}

    def get_stats(self):
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self.labels_df),
            'label_distribution': {}
        }
        
        for col in self.label_columns:
            value_counts = self.labels_df[col].value_counts()
            stats['label_distribution'][col] = {
                str(k): int(v) for k, v in value_counts.items()
            }
            
        return stats

    def check_data_types(self):
        """Print data type information for debugging."""
        print("\nDataset Info:")
        print(f"Number of samples: {len(self.labels_df)}")
        print("\nColumn Types:")
        print(self.labels_df[self.label_columns].dtypes)
        print("\nSample Values:")
        print(self.labels_df[self.label_columns].head())