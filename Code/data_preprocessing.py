import SimpleITK as sitk
import numpy as np 
# import matplotlib.pyplot as plt 
import glob
import os
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import random
import segmentation_models_pytorch as smp
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms.functional as TF
from tqdm import tqdm
import concurrent.futures



# Reading Data
def read_image(path):
    """
    Read the image from the provided URL and convert it to an array using SimpleITK.

    @param path: get a url path
    @return: the array of image
    """
    image = sitk.ReadImage(path) 
    image_array = sitk.GetArrayFromImage(image)

    return image_array


def check_shape(image_paths, mask_paths):
    """
    Check the shapes of the images with their related masks and return an error in case they do not match.

    @param image_paths: list of image paths
    @param mask_paths: list of mask paths
    """
    print("Checking shapes in process ...")
    for i in tqdm(range(len(image_paths))):
        image = read_image(image_paths[i])
        mask = read_image(mask_paths[i])

        assert image.shape == mask.shape, f"case: {i}, image shape does not match with the related mask"
    
    print("Checking images and masks shapes successfully completed")

def convert_3d_to_2d(dimension, image_paths, root, directory):
    """
    Converting 3D images to 2D slices

    @param dimension: the fixed axis
    @param images_path: list of image paths
    @param root: root url
    @param directory: the name of folder that the new data will be stored in
    """
    print("converting 3D images to 2D slices ...")
    for i in tqdm(range(len(image_paths))):
        image = read_image(image_paths[i])
        for j in range(image.shape[dimension]):
            if dimension == 2:
                image_slice = sitk.GetImageFromArray(image[:,:,j])
            elif dimension == 1:
                image_slice = sitk.GetImageFromArray(image[:,j,:])
            elif dimension == 0:
                image_slice = sitk.GetImageFromArray(image[j,:,:])

            dir = f'{directory}_d{dimension}'
            folder = "case_" + (5 - len(str(i))) * "0" + str(i)
            path = os.path.join(root + "/" + folder, dir)
            if not os.path.exists(path):
                os.makedirs(path)
            filename = f'{directory}_slice_{(5 - len(str(j))) * "0" + str(j)}.nii.gz'
            filepath = os.path.join(path, filename)
            print(filepath)
            sitk.WriteImage(image_slice, filepath)


def convert_3d_to_2d_threaded(dimension, image_paths, root, directory):
    """
    Converting 3D images to 2D slices using multithreading.

    @param dimension: the fixed axis
    @param images_path: list of image paths
    @param root: root url
    @param directory: the name of folder that the new data will be stored in
    """
    print("Converting 3D images to 2D slices using multithreading ...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in tqdm(range(len(image_paths))):
            image = read_image(image_paths[i])
            for j in range(image.shape[dimension]):
                if dimension == 2:
                    image_slice = sitk.GetImageFromArray(image[:, :, j])
                elif dimension == 1:
                    image_slice = sitk.GetImageFromArray(image[:, j, :])
                elif dimension == 0:
                    image_slice = sitk.GetImageFromArray(image[j, :, :])

                dir = f'{directory}_d{dimension}'
                folder = "case_" + (5 - len(str(i))) * "0" + str(i)
                path = os.path.join(root + "/" + folder, dir)
                if not os.path.exists(path):
                    os.makedirs(path)
                filename = f'{directory}_slice_{(5 - len(str(j))) * "0" + str(j)}.nii.gz'
                filepath = os.path.join(path, filename)
                # print(filepath)

                futures.append(executor.submit(sitk.WriteImage, image_slice, filepath))

        # Wait for all futures to complete
        concurrent.futures.wait(futures)


## Reading the path of the original images
root = "kits23/dataset"
image_paths = sorted(glob.glob(f'{root}/**/imaging.nii.gz', recursive=True))
mask_paths = sorted(glob.glob(f'{root}/**/segmentation.nii.gz', recursive=True))

## Data Checking
assert len(image_paths) == len(mask_paths), "The number of labels does not match with the masks"

# check_shape(image_paths, mask_paths)

## Data Preprocessing


# convert_3d_to_2d_threaded(2, image_paths, root, "kidney_images")

convert_3d_to_2d_threaded(2, mask_paths, root, "kidney_annotations")




