from torch import nn
import torch
from torchvision.models import inception_v3
import torch.nn.functional as F

size = (1, 224, 224)

class ImageColorNet(nn.Module):
    def __init__(self, batch_size):
        super(ImageColorNet, self).__init__()

        self.batch_size = batch_size

        # encoder - input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # add layers
        
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool9 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv_10 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv_11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv_12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv_13 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

        self.model_inceptionv3 = inception_v3(pretrained=True, aux_logits=False)
        self.linear1 = nn.Linear(2048, 128)
        self.linear2 = nn.Linear(128, 1)

        for i, param in self.model_inceptionv3.named_parameters():
              param.requires_grad = False

        num_ftrs = self.model_inceptionv3.fc.in_features
        self.model_inceptionv3.fc = nn.Linear(num_ftrs, num_ftrs*128)

        for name, child in self.model_inceptionv3.named_children():
            for params in child.parameters():
                params.requires_grad = False
                
        self.conv_15 = nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1)
        self.conv_16 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=1)
        self.conv_17 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_18 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_19 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv_20 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv_21 = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=(1, 1))
        self.conv_22 = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=2, padding=1)

    def forward(self, output, output_inception):
        output = output.view(self.batch_size,1,224,224).float()
        
        x = F.relu(self.conv1(output))
        x = self.pool1(x)
        conv2 = F.relu(self.conv2(x))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))
        
        layer4 = conv2 + conv4
        
        x = F.relu(self.conv5(layer4))
        x = self.pool5(x)
        conv6 = F.relu(self.conv6(x))
        conv7 = F.relu(self.conv7(conv6))
        conv8 = F.relu(self.conv8(conv7))
        
        layer8 = conv6 + conv8
        
        x = F.relu(self.conv9(layer8))
        x = self.pool9(x)
        conv_10 = F.relu(self.conv_10(x))
        conv_11 = F.relu(self.conv_11(conv_10))
        conv_12 = F.relu(self.conv_12(conv_11))
        
        layer_12 = conv_10 + conv_12
        
        x = F.relu(self.conv_13(layer_12))
        
        inception_output = self.model_inceptionv3.forward(output_inception.view(self.batch_size,3,299,299))
        inception_output = self.linear1(inception_output.view(self.batch_size, 128, 2048))
        inception_output = self.linear2(inception_output.view(self.batch_size, 128, 128))
        
        inception_output = inception_output.view(self.batch_size, 128)
        inception_output = inception_output.repeat(28, 28)
        
        x = torch.cat([inception_output.view(self.batch_size, 128, 28, 28), x], dim=1)
        
        x = self.conv_15(x)
        x = F.relu(self.conv_16(x))
        x = F.relu(self.conv_17(x))
        x = F.relu(self.conv_18(x))
        x = F.relu(self.conv_19(x))
        x = F.relu(self.conv_20(x))
        x = F.relu(self.conv_21(x))
        x = F.relu(self.conv_22(x))
        
        return x