import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# The implementations of the base FCN-32, FCN-16, and FCN-8 architectures have been
# adapted and modified from the pytorch-fcn library (https://github.com/wkentaro/pytorch-fcn/tree/main).

class FCN32(nn.Module):
    def __init__(self, number_of_classes, pixels_cut):
        super(FCN32, self).__init__()

        self.pixels_cut = pixels_cut

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=100)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 =nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.drop5 = nn.Dropout2d()

        self.fc6 = nn.Conv2d(256, 4096, kernel_size=6)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, number_of_classes, kernel_size=1)
        self.upscore = nn.ConvTranspose2d(number_of_classes, number_of_classes, 63, stride=32, bias=False)

    def forward(self, x):
        h = x.float()
        h = self.relu1(self.conv1(h))
        h = self.pool1(h)

        h = self.relu2(self.conv2(h))
        h = self.pool2(h)

        h = self.relu3(self.conv3(h))

        h = self.relu4(self.conv4(h))

        h = self.relu5(self.conv5(h))
        h = self.pool5(h)
        h= self.drop5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)

        h = h[:, :, self.pixels_cut:self.pixels_cut + x.size()[2], self.pixels_cut:self.pixels_cut + x.size()[3]].contiguous()
        
        return h


class FCN16(nn.Module):
    def __init__(self, number_of_classes, pixels_cut):
        super(FCN16, self).__init__()

        self.pixels_cut = pixels_cut

        self.pool3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=100),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.pool4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.score_fr = nn.Conv2d(4096, number_of_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, number_of_classes, kernel_size=1)

        self.upsample2 = nn.ConvTranspose2d(number_of_classes, number_of_classes, kernel_size=4, stride=2, bias=False)
        self.upsample16 = nn.ConvTranspose2d(number_of_classes, number_of_classes, kernel_size=32, stride=16, bias=False)
        
    def forward(self, x):
        h = x.float()
        h = self.pool3(h)

        h = self.pool4(h)

        pool4 = h

        conv7 = self.conv7(h)
        
        conv7_score = self.score_fr(conv7)
        pool4_score = self.score_pool4(pool4)

        upsampled_conv7 = self.upsample2(conv7_score) 

        cut_pool4 = pool4_score[:, :, 5:5 + upsampled_conv7.size()[2], 5:5 + upsampled_conv7.size()[3]]
        upscore_temp = upsampled_conv7 + cut_pool4 

        return self.upsample16(upscore_temp)[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]]

class FCN8(nn.Module):
    def __init__(self, number_of_classes, pixels_cut):
        super(FCN8, self).__init__()

        self.pixels_cut = pixels_cut

        self.pool3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=100),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.pool4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.score_fr = nn.Conv2d(4096, number_of_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, number_of_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, number_of_classes, kernel_size=1)

        self.upsample2 = nn.ConvTranspose2d(number_of_classes, number_of_classes, kernel_size=4, stride=2, bias=False)
        self.upsample_pool4 = nn.ConvTranspose2d(number_of_classes, number_of_classes, kernel_size=4, stride=2, bias=False)
        self.upsample8 = nn.ConvTranspose2d(number_of_classes, number_of_classes, kernel_size=16, stride=8, bias=False)
        
    def forward(self, x):
        h = x.float()
        h = self.pool3(h)
        
        pool3 = h

        h = self.pool4(h)

        pool4 = h

        conv7 = self.conv7(h)
        
        conv7_score = self.score_fr(conv7)
        pool3_score = self.score_pool3(pool3)
        pool4_score = self.score_pool4(pool4)

        upsampled_conv7 = self.upsample2(conv7_score)
        cut_pool4 = pool4_score[:, :, 5:5 + upsampled_conv7.size()[2], 5:5 + upsampled_conv7.size()[3]] 

        upscore_temp = self.upsample_pool4(upsampled_conv7 + cut_pool4)

        cut_pool3 = pool3_score[:, :, 9:9 + upscore_temp.size()[2], 9:9 + upscore_temp.size()[3]]

        upscore_temp = upscore_temp + cut_pool3

        return self.upsample8(upscore_temp)[:, :, 36:36 + x.size()[2], 36:36 + x.size()[3]]


def _data_parallel_state_dict_to_normal(state_dict):
    return {key.replace("module.", ""): value for key, value in state_dict.items()}

class CombinedNet6Channels(nn.Module):
    def __init__(self, number_of_classes, pixels_cut, fcn_32_path, fcn_16_path, fcn_8_path):
        super(CombinedNet6Channels, self).__init__()

        device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.model_fcn32 = FCN32(number_of_classes=number_of_classes, pixels_cut=pixels_cut).to(device)
        self.model_fcn32.load_state_dict(_data_parallel_state_dict_to_normal(torch.load(fcn_32_path)))
        self.model_fcn32.eval()

        self.model_fcn16 = FCN16(number_of_classes=number_of_classes, pixels_cut=pixels_cut).to(device)
        self.model_fcn16.load_state_dict(_data_parallel_state_dict_to_normal(torch.load(fcn_16_path)))
        self.model_fcn16.eval()

        self.model_fcn8 = FCN8(number_of_classes=number_of_classes, pixels_cut=pixels_cut).to(device)
        self.model_fcn8.load_state_dict(_data_parallel_state_dict_to_normal(torch.load(fcn_8_path)))
        self.model_fcn8.eval()

        self.pixels_cut = pixels_cut

        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=1 + 1 + 1 + 3, out_channels=64, kernel_size=11, stride=4, padding=100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Dropout2d(),

            nn.Conv2d(256, 4096, kernel_size=6),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.score_fr = nn.Conv2d(4096, number_of_classes, kernel_size=1)
        self.upscore = nn.ConvTranspose2d(number_of_classes, number_of_classes, 63, stride=32, bias=False)

    def forward(self, x):
        images = x

        with torch.no_grad():
            fcn32_outputs = torch.sigmoid(self.model_fcn32(images))
            fcn16_outputs = torch.sigmoid(self.model_fcn16(images))
            fcn8_outputs = torch.sigmoid(self.model_fcn8(images))

        h = torch.cat((fcn32_outputs, fcn16_outputs, fcn8_outputs, images.float()), 1)

        h = self.convolution(h)

        h = self.score_fr(h)

        h = self.upscore(h)

        h = h[:, :, self.pixels_cut:self.pixels_cut + x.size()[2], self.pixels_cut:self.pixels_cut + x.size()[3]]
        
        return h


class CombinedNet5Channels3216(nn.Module):
    def __init__(self, number_of_classes, pixels_cut, fcn_32_path, fcn_16_path):
        super(CombinedNet5Channels3216, self).__init__()

        device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.model_fcn32 = FCN32(number_of_classes=number_of_classes, pixels_cut=pixels_cut).to(device)
        self.model_fcn32.load_state_dict(_data_parallel_state_dict_to_normal(torch.load(fcn_32_path)))
        self.model_fcn32.eval()

        self.model_fcn16 = FCN16(number_of_classes=number_of_classes, pixels_cut=pixels_cut).to(device)
        self.model_fcn16.load_state_dict(_data_parallel_state_dict_to_normal(torch.load(fcn_16_path)))
        self.model_fcn16.eval()

        self.pixels_cut = pixels_cut

        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=1 + 1 + 3, out_channels=64, kernel_size=11, stride=4, padding=100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True), 
            nn.Dropout2d(),

            nn.Conv2d(256, 4096, kernel_size=6),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.score_fr = nn.Conv2d(4096, number_of_classes, kernel_size=1)
        self.upscore = nn.ConvTranspose2d(number_of_classes, number_of_classes, 63, stride=32, bias=False)

    def forward(self, x):
        images = x

        with torch.no_grad():
            fcn32_outputs = torch.sigmoid(self.model_fcn32(images))
            fcn16_outputs = torch.sigmoid(self.model_fcn16(images))

        h = torch.cat((fcn32_outputs, fcn16_outputs, images.float()), 1)

        h = self.convolution(h)

        h = self.score_fr(h)

        h = self.upscore(h)

        h = h[:, :, self.pixels_cut:self.pixels_cut + x.size()[2], self.pixels_cut:self.pixels_cut + x.size()[3]]
        
        return h

class CombinedNet5Channels168(nn.Module):
    def __init__(self, number_of_classes, pixels_cut, fcn_16_path, fcn_8_path):
        super(CombinedNet5Channels168, self).__init__()

        device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.model_fcn16 = FCN16(number_of_classes=number_of_classes, pixels_cut=pixels_cut).to(device)
        self.model_fcn16.load_state_dict(_data_parallel_state_dict_to_normal(torch.load(fcn_16_path)))
        self.model_fcn16.eval()

        self.model_fcn8 = FCN8(number_of_classes=number_of_classes, pixels_cut=pixels_cut).to(device)
        self.model_fcn8.load_state_dict(_data_parallel_state_dict_to_normal(torch.load(fcn_8_path)))
        self.model_fcn8.eval()

        self.pixels_cut = pixels_cut

        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=1 + 1 + 3, out_channels=64, kernel_size=11, stride=4, padding=100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True), 
            nn.Dropout2d(),

            nn.Conv2d(256, 4096, kernel_size=6),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.score_fr = nn.Conv2d(4096, number_of_classes, kernel_size=1)
        self.upscore = nn.ConvTranspose2d(number_of_classes, number_of_classes, 63, stride=32, bias=False)

    def forward(self, x):
        images = x

        with torch.no_grad():
            fcn16_outputs = torch.sigmoid(self.model_fcn16(images))
            fcn8_outputs = torch.sigmoid(self.model_fcn8(images))

        h = torch.cat((fcn16_outputs, fcn8_outputs, images.float()), 1)

        h = self.convolution(h)

        h = self.score_fr(h)

        h = self.upscore(h)

        h = h[:, :, self.pixels_cut:self.pixels_cut + x.size()[2], self.pixels_cut:self.pixels_cut + x.size()[3]]
        
        return h
    

class CombinedNet5Channels328(nn.Module):
    def __init__(self, number_of_classes, pixels_cut, fcn_32_path, fcn_8_path):
        super(CombinedNet5Channels328, self).__init__()

        device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.model_fcn32 = FCN16(number_of_classes=number_of_classes, pixels_cut=pixels_cut).to(device)
        self.model_fcn32.load_state_dict(_data_parallel_state_dict_to_normal(torch.load(fcn_32_path)))
        self.model_fcn32.eval()

        self.model_fcn8 = FCN8(number_of_classes=number_of_classes, pixels_cut=pixels_cut).to(device)
        self.model_fcn8.load_state_dict(_data_parallel_state_dict_to_normal(torch.load(fcn_8_path)))
        self.model_fcn8.eval()

        self.pixels_cut = pixels_cut

        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=1 + 1 + 3, out_channels=64, kernel_size=11, stride=4, padding=100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True), 
            nn.Dropout2d(),

            nn.Conv2d(256, 4096, kernel_size=6),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.score_fr = nn.Conv2d(4096, number_of_classes, kernel_size=1)
        self.upscore = nn.ConvTranspose2d(number_of_classes, number_of_classes, 63, stride=32, bias=False)

    def forward(self, x):
        images = x

        with torch.no_grad():
            fcn32_outputs = torch.sigmoid(self.model_fcn32(images))
            fcn8_outputs = torch.sigmoid(self.model_fcn8(images))

        h = torch.cat((fcn32_outputs, fcn8_outputs, images.float()), 1)

        h = self.convolution(h)

        h = self.score_fr(h)

        h = self.upscore(h)

        h = h[:, :, self.pixels_cut:self.pixels_cut + x.size()[2], self.pixels_cut:self.pixels_cut + x.size()[3]]
        
        return h
