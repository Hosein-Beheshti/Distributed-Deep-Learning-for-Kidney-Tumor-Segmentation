import SimpleITK as sitk
import numpy as np 
import matplotlib.pyplot as plt 
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
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from data_loading.data_reader import read_image, create_df
from data_loading.dataset import ImageDataset
from model.model import UNet
from training.train import train_model
import argparse



def visualize(image):
  plt.figure()
  # plt.imshow(image.detach().squeeze(), cmap='gray')
  plt.imshow(image.squeeze(), cmap='gray')

  plt.axis('off')
  plt.show()



if __name__ == "__main__":

     # Create argument parser
    parser = argparse.ArgumentParser(description="Your script description here.")

    # Add optional arguments
    parser.add_argument("--dp", type=bool, default=False, help="Description of data_parallel.")
    parser.add_argument("--ddp", type=bool, default=False, help="Description of ddp_flag.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs that will be used for data parallel")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size that will be used in training")
    


    # Parse command-line arguments
    args = parser.parse_args()

    data_parallel = args.dp
    ddp_flag = args.ddp
    batch_size = args.batch_size
    num_gpus = args.num_gpus

    print("Data Parallel:", data_parallel, "DDP:", ddp_flag, "batch_size:", batch_size, "num_gpus:", num_gpus)


    root = "kits23/dataset"
    image_paths = sorted(glob.glob(f'{root}/**/kidney_images_d2/*.nii.gz', recursive=True))
    mask_paths = sorted(glob.glob(f'{root}/**/kidney_annotations_d2/*.nii.gz', recursive=True))

    # Create a dataframe with GID and IID
    path_df = create_df(image_paths, mask_paths)

    # Spliting train, validation and test dataset
    train_df = path_df[path_df['GID'] < 400].reset_index(drop=True)
    val_df = path_df[(path_df['GID'] >= 400) & (path_df['GID'] < 444)].reset_index(drop=True)
    test_df = path_df[path_df['GID'] >= 444].reset_index(drop=True)

    print(len(train_df),len(test_df),len(val_df))  


    # Creating augmentations pipeline
    mean = 0.6519684854829247
    std = 4.730642029234591

    # Define the min and max ranges for clipping the image
    min_range = 0
    max_range = 200

    # Define the transformation pipeline
    image_transform = transforms.Compose([
        transforms.Lambda(lambda x: np.clip(x, min_range, max_range)),  # Clip the image
        transforms.Normalize(mean=mean, std=std) # Normalize the tensor
    ])

    # Define the transformation pipeline - this will be used for both image and mask
    general_transform = A.Compose([
        # A.PadIfNeeded(0, 1024, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Resize(height=512, width=512),
        ToTensorV2()
        # transforms.RandomHorizontalFlip(0.2)
    ])
    random_transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Rotate(limit=(-30, 30), border_mode=0, p=0.3),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(p=0.5),
        A.RandomCrop(width=380, height=380, p=0.5)
        # transforms.RandomHorizontalFlip(0.2)
    ])
    
    train_dataset = ImageDataset(path_df=train_df, image_transform=image_transform, general_transform=general_transform, random_transform=random_transform)
    val_dataset = ImageDataset(path_df=val_df, image_transform=image_transform, general_transform=general_transform)
    test_dataset = ImageDataset(path_df=test_df, image_transform=image_transform, general_transform=general_transform, return_id=True)

    # x = train_dataset[15200]
    # visualize(x[0])
    # visualize(x[1])


    if data_parallel:
        if num_gpus == 2:
            gpu_devices = [0, 1]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))
        elif num_gpus == 4:
            gpu_devices = [0, 1, 2, 3]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if ddp_flag:
        print("setting up ddp")
        init_process_group("nccl")
        rank = dist.get_rank()
        print(rank)
        device = rank % torch.cuda.device_count()


    # Creating data loaders
    num_workers = 4
        
    if ddp_flag:
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle = False, num_workers=num_workers, sampler=DistributedSampler(train_dataset))
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False, num_workers=num_workers, sampler=DistributedSampler(test_dataset))
        val_dataloader = DataLoader(val_dataset, batch_size, shuffle = False, num_workers=num_workers,sampler=DistributedSampler(val_dataset))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size, shuffle = False, num_workers=num_workers)

    
    pretrained_weights_path = '/home/hoseinb/scratch/ESNF619/hoseinCC.git/ResNet50.pt'
    unet_model = UNet(classes=4, pretrained_weights_path=pretrained_weights_path)
    model = unet_model.get_model().to(device) 


    # Data parallel over GPUs
    if torch.cuda.device_count() > 1 and data_parallel:
        print("Using", torch.cuda.device_count(), "GPUs for training.")
        model = torch.nn.DataParallel(model)




    # Distributed Learning

    if ddp_flag:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        # gpu_id = int(os.environ["LOCAL_RANK"])
        model = model.to(device)
        model = DDP(model, device_ids=[device])



    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs for training.")
    #     model = torch.nn.DataParallel(model)
    

    start_time = time.time()

    train_model(model, device, unet_model.classes, train_dataloader, val_dataloader, 50, None)

    if ddp_flag:
        destroy_process_group()

    end_time = time.time()

    training_time = end_time - start_time

    # Convert the training time to hours, minutes, and seconds for better readability
    training_time_hours = int(training_time // 3600)
    training_time_minutes = int((training_time % 3600) // 60)
    training_time_seconds = int(training_time % 60)

    print(f"Total training time: {training_time_hours} hours, {training_time_minutes} minutes, {training_time_seconds} seconds")