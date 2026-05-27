import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

class MedicalImageDataset(Dataset):
    """Dataset for medical images (CXR or Sputum)"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(image_size, is_train=True):
    """Get image transformations"""
    if is_train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

def load_medical_data(data_dir, dataset_type="cxr"):
    """Load medical image dataset"""
    image_paths = []
    labels = []
    
    # Assuming structure: data_dir/class_name/image_files
    for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(class_idx)
    
    return np.array(image_paths), np.array(labels)

def create_client_dataloaders(client_data_dict, batch_size, image_size):
    """Create dataloaders for each client"""
    client_loaders = {}
    
    for client_id, (train_paths, train_labels, val_paths, val_labels) in client_data_dict.items():
        # Training loader
        train_dataset = MedicalImageDataset(
            train_paths, train_labels,
            transform=get_transforms(image_size, is_train=True)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )
        
        # Validation loader
        val_dataset = MedicalImageDataset(
            val_paths, val_labels,
            transform=get_transforms(image_size, is_train=False)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        client_loaders[client_id] = {
            'train': train_loader,
            'val': val_loader
        }
    
    return client_loaders