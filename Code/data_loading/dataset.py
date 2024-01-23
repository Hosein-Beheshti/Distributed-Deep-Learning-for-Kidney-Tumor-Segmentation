import SimpleITK as sitk
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt 
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms.functional as TF
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, path_df, image_transform=None, general_transform=None, random_transform=None, return_id=None):
        self.image_transform = image_transform
        self.general_transform = general_transform
        self.random_transform = random_transform
        self.return_id = return_id
        self.path_df = path_df
                
    def __len__(self):
        return len(self.path_df)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.path_df['Image'][idx])
        image_array = sitk.GetArrayFromImage(image)
        mask = sitk.ReadImage(self.path_df['Mask'][idx]) 
        mask_array = sitk.GetArrayFromImage(mask)
        
        if self.random_transform:
            transformed = self.random_transform(image=image_array, mask=mask_array)
            image_array = transformed["image"]
            mask_array = transformed["mask"]
            
        if self.general_transform:
            transformed = self.general_transform(image=image_array, mask=mask_array)
            image_array = transformed["image"]
            mask_array = transformed["mask"]
            mask_array = mask_array.unsqueeze(0)

        if self.image_transform:
            image_array = self.image_transform(image_array)
        
        if self.return_id:
            return image_array, mask_array, self.path_df['GID'][idx], self.path_df['IID'][idx]

        return image_array, mask_array