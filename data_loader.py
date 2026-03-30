import os
import glob
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
# Define our universal master classes
MASTER_CLASSES = {
    0: 'Intact',
    1: 'Crack',
    2: 'Spalling/Severe Damage'
}

def get_drone_augmentations(img_size=640, is_train=True):
    """
    Advanced Data Augmentations mimicking Drone structural inspections using Albumentations.
    Injects dynamic variations: Glare, motion blur from rotors, perspective shifts.
    """
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # Simulate real-world drone conditions
            A.OneOf([
                A.MotionBlur(p=0.8, blur_limit=7),
                A.GaussNoise(p=0.6, var_limit=(10.0, 50.0)),
                A.Defocus(p=0.5, radius=(1, 5), alias_blur=(0.1, 0.5))
            ], p=0.4),
            A.OneOf([
                A.RandomSunFlare(p=0.3, flare_roi=(0, 0, 1, 0.5), src_radius=150),
                A.RandomShadow(p=0.3, num_shadows_lower=1, num_shadows_upper=2)
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

class SDNET2018Dataset(Dataset):
    """
    Custom PyTorch Dataset for SDNET2018.
    Typically contains structures like /C (Crack) and /U (Uncracked).
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Unify SDNET classification scheme into master classes
        self.label_map = {
            'U': 0, # Intact
            'C': 1  # Crack
            # (SDNET mostly focuses on discrete crack vs uncracked states)
        }
        self._load_dataset()
        
    def _load_dataset(self):
        if not self.root_dir.exists():
            print(f"Warning: SDNET2018 directory {self.root_dir} not found.")
            return
            
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in self.label_map:
                label_idx = self.label_map[class_dir.name]
                for ext in ('*.jpg', '*.jpeg', '*.png'):
                    for img_path in class_dir.glob(ext):
                        self.image_paths.append(str(img_path))
                        self.labels.append(label_idx)
                        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        target = self.labels[idx]
        
        if self.transform:
            # Albumentations expects a numpy array, not PIL
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = augmented['image']
            
        return image, torch.tensor(target, dtype=torch.long)

class PHIDataset(Dataset):
    """
    Custom PyTorch Dataset for PEER Hub ImageNet (PHI).
    PHI includes multiple structural damage states.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Unify PHI classification scheme into master classes
        self.label_map = {
            'undamaged': 0,         # Intact
            'crack': 1,             # Crack
            'spalling': 2,          # Spalling/Severe Damage
            'severe_damage': 2      # Spalling/Severe Damage
        }
        self._load_dataset()
        
    def _load_dataset(self):
        if not self.root_dir.exists():
            print(f"Warning: PHI directory {self.root_dir} not found.")
            return

        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in self.label_map:
                label_idx = self.label_map[class_dir.name]
                for ext in ('*.jpg', '*.jpeg', '*.png'):
                    for img_path in class_dir.glob(ext):
                        self.image_paths.append(str(img_path))
                        self.labels.append(label_idx)
                        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        target = self.labels[idx]
        
        if self.transform:
            # Albumentations expects a numpy array, not PIL
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = augmented['image']
            
        return image, torch.tensor(target, dtype=torch.long)

def get_unified_dataloader(sdnet_path, phi_path, batch_size=32, img_size=640, num_workers=4, is_train=True):
    """
    Merges SDNET2018 and PHI datasets into a single DataLoader.
    Applies unified dynamic drone augmentations and label mappings.
    """
    transform = get_drone_augmentations(img_size, is_train=is_train)
    
    sdnet_ds = SDNET2018Dataset(root_dir=sdnet_path, transform=transform)
    phi_ds = PHIDataset(root_dir=phi_path, transform=transform)
    
    # Merge using torch.utils.data.ConcatDataset
    combined_dataset = ConcatDataset([sdnet_ds, phi_ds])
    
    if len(combined_dataset) == 0:
        print("Warning: Combined dataset is empty. Ensure data paths are correct.")
        
    # Standard PyTorch DataLoader
    dataloader = DataLoader(combined_dataset, 
                            batch_size=batch_size, 
                            shuffle=is_train, 
                            num_workers=num_workers)
    
    return dataloader, combined_dataset

if __name__ == "__main__":
    # Example Usage:
    sdnet_dir = "data/sdnet2018"
    phi_dir = "data/phi"
    
    loader, dataset = get_unified_dataloader(sdnet_dir, phi_dir)
    print(f"Total unified samples: {len(dataset)}")
