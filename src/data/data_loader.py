import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ReflexDataset(Dataset):
    """Dataset class for RefleX X-ray diffraction images.
    
    This dataset handles multi-label classification with 7 possible labels:
    Ice ring, Diffuse Scattering, Background Ring, Non-uniform Detector, 
    Loop Scattering, Strong Background, and Artifact.
    
    Args:
        root_dir (str): Directory containing the images
        labels_file (str): Path to labels CSV file
        transform (callable, optional): Optional transform to be applied to images
    """
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Read labels file
        self.labels_df = pd.read_csv(labels_file)
        
        # Define the label columns
        self.label_columns = [
            'Ice ring', 'Diffuse Scattering', 'Background Ring',
            'Non-uniform Detector', 'Loop Scattering', 'Strong Background',
            'Artifact'
        ]

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        """Returns a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            dict: Contains 'image' tensor and 'labels' tensor
        """
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
            
        # Get image path and load image
        img_name = self.labels_df.iloc[idx]['image_name']
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)
        
        # Get labels
        labels = self.labels_df.iloc[idx][self.label_columns].values
        labels = torch.FloatTensor(labels)  # Use float for BCE loss
        
        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'labels': labels}