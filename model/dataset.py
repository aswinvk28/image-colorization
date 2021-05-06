import torch
from torchvision import datasets, transforms
from glob import glob
import numpy as np
from PIL import Image

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])
color_transform = transforms.Compose([transforms.Resize((225,225)),
                                transforms.ToTensor()])

inception_transform = transforms.Compose([transforms.Resize((299,299)),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor()])

dataset = datasets.ImageFolder(
  "E:\\Drives-Linux-ubuntu-2020\\home\\aswin\\Documents\\Deep-Learning-Projects\\open_images_dataset\\dataset\\train", 
  transform=transform)
inception_dataset = datasets.ImageFolder(
  "E:\\Drives-Linux-ubuntu-2020\\home\\aswin\\Documents\\Deep-Learning-Projects\\open_images_dataset\\dataset\\train", 
  transform=inception_transform)
color_dataset = datasets.ImageFolder(
  "E:\\Drives-Linux-ubuntu-2020\\home\\aswin\\Documents\\Deep-Learning-Projects\\open_images_dataset\\dataset\\train", 
  transform=color_transform)

val_dataset = datasets.ImageFolder(
  "E:\\Drives-Linux-ubuntu-2020\\home\\aswin\\Documents\\Deep-Learning-Projects\\open_images_dataset\\dataset\\validation", 
  transform=transform)
val_inception_dataset = datasets.ImageFolder(
  "E:\\Drives-Linux-ubuntu-2020\\home\\aswin\\Documents\\Deep-Learning-Projects\\open_images_dataset\\dataset\\validation", 
  transform=inception_transform)
val_color_dataset = datasets.ImageFolder(
  "E:\\Drives-Linux-ubuntu-2020\\home\\aswin\\Documents\\Deep-Learning-Projects\\open_images_dataset\\dataset\\validation", 
  transform=color_transform)