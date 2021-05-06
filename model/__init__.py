import os
import torch
import numpy as np
from tqdm import tqdm
import gc
from torch.autograd import Variable
from torchvision import transforms, models
from torch import nn, optim
import cv2
import math
import threading
from PIL import Image
import cv2
from .dataset import dataset, inception_dataset, color_dataset, val_dataset, val_color_dataset, val_inception_dataset
from .persistence import load_model, save_model
from .model import ImageColorNet
from pytorch_model_summary import summary
import pytorch_colors as colors

def run_model(start_epoch, diff_epoch):
  clip = 5

  batch_size = 32

  net = ImageColorNet(batch_size)
  print(summary(net, torch.zeros(batch_size,1,224,224), torch.zeros(batch_size,3,299,299), show_input=False))
  net_optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8)

  torch.autograd.set_detect_anomaly(True)
  dataloader = enumerate(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False))
  inception_dataloader = enumerate(torch.utils.data.DataLoader(inception_dataset, batch_size=batch_size, shuffle=False))
  color_dataloader = enumerate(torch.utils.data.DataLoader(color_dataset, batch_size=batch_size, shuffle=False))
  val_dataloader = enumerate(torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False))
  val_inception_dataloader = enumerate(torch.utils.data.DataLoader(val_inception_dataset, batch_size=batch_size, shuffle=False))
  val_color_dataloader = enumerate(torch.utils.data.DataLoader(val_color_dataset, batch_size=batch_size, shuffle=False))

  criterion = torch.nn.MSELoss(reduction='sum')
  val_criterion = torch.nn.MSELoss(reduction='mean')

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  for epoch in tqdm(range(start_epoch, start_epoch + diff_epoch)):

    # load network from checkpoint
    if os.path.isfile('checkpoints/img_color-'+str(epoch)+'_'+str(0)+'.checkpoint.pth'):
      net = load_model(net, net_optimizer, epoch, 0)

    print("Training Epoch: ", epoch)
    print("Batch Size: ", batch_size)
    for ii, data in color_dataloader:
      net.train()
      color_images, color_labels = data
      i, data = next(iter(dataloader))
      images, labels = data
      i, inception_data = next(iter(inception_dataloader))
      inception_images, inception_labels = inception_data

      output = net.forward(images, inception_images)
      l2_reg = None
      reg_lambda = 0.5
      for i, param in net.named_parameters():
        if ("conv3.weight" in i) or \
        ("conv4.weight" in i) or ("conv5.weight" in i) or \
        ("conv6.weight" in i) or ("conv7.weight" in i) or \
        ("conv8.weight" in i) or ("conv9.weight" in i) or \
        ("conv_12.weight" in i) or ("conv_13.weight" in i) or \
        ("conv_18.weight" in i) or ("conv_19.weight" in i) or \
        ("conv_20.weight" in i) or ("conv_21.weight" in i) or ("conv_22.weight" in i):
          if l2_reg is None:
            l2_reg = param.norm(2)**2
          else:
            l2_reg = l2_reg + param.norm(2)**2
      lab_images = colors.rgb_to_lab(color_images) / 255
      loss = criterion(output, lab_images[:,1:,:,:]) + l2_reg * reg_lambda
      loss.backward()
      print("Epoch: ", epoch, ", Batch: ", ii,  ", Loss: ", loss.item())
      nn.utils.clip_grad_norm_(net.parameters(), clip)
      net_optimizer.step()
      net_optimizer.zero_grad()

      if(ii % 10 == 0):
        net.eval()
        with torch.no_grad():
          i, val_data = next(iter(val_dataloader))
          val_images, val_labels = val_data
          i, val_inception_data = next(iter(val_inception_dataloader))
          val_inception_images, val_inception_labels = val_inception_data
          i, val_color_data = next(iter(val_color_dataloader))
          val_color_images, val_color_labels = val_color_data
          val_lab_images = colors.rgb_to_lab(val_color_images) / 255
          val_output = net.forward(val_images, val_inception_images)
          val_loss = criterion(val_output, val_lab_images[:,1:,:,:])
          val_accuracy = val_criterion(val_output, val_lab_images[:,1:,:,:])
          print("Epoch: ", epoch, ", Batch: ", ii,  ", Val Loss: ", val_loss.item(), ", Mean Accuracy: ", val_accuracy.item())

          save_model(net, net_optimizer, epoch, ii)
          print("--Model Saved--")