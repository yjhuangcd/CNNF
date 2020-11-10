import torch
import torch.nn as nn
import torch.nn.functional as F
import cnnf.layers as layers
import logging
import os
import torch.optim as optim
import numpy as np
import math
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import shutil
from tensorboardX import SummaryWriter

class CNNF(nn.Module):
    """ CNNF on an architecture with 4 Convs and 3 FCs. """
    
    def __init__(self, num_classes, ind=0, cycles=2, res_param=0.1):
        super(CNNF, self).__init__()
        
        self.ind = ind
        self.res_param = res_param
        self.cycles = cycles
        # 1st conv before any network block
        self.conv1 = layers.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.ins1 = layers.InsNorm()
        self.ins1_bias = layers.Bias((1,16,1,1))
        self.relu1 = layers.resReLU(res_param)
        self.conv2 = layers.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.ins2 = layers.InsNorm()
        self.ins2_bias = layers.Bias((1,32,1,1))
        self.relu2 = layers.resReLU(res_param)
        self.conv3 = layers.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.ins3 = layers.InsNorm()
        self.ins3_bias = layers.Bias((1,32,1,1))
        self.relu3 = layers.resReLU(res_param)
        self.maxpool3 = layers.MaxPool2d(2)
        self.conv4 = layers.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.ins4 = layers.InsNorm()
        self.ins4_bias = layers.Bias((1,64,1,1))
        self.relu4 = layers.resReLU(res_param)
        self.maxpool4 = layers.MaxPool2d(2)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Linear(3136, 1000)
        self.fc1_bias = layers.Bias((1,1000))
        self.relu5 = layers.resReLU(res_param)  
        self.fc2 = layers.Linear(1000, 128)
        self.fc2_bias = layers.Bias((1,128))
        self.relu6 = layers.resReLU(res_param)  
        self.fc3 = layers.Linear(128, num_classes)
        self.fc3_bias = layers.Bias((1,num_classes))

    def forward(self, out, step='forward', first=True, inter=False, inter_recon=False):     
        if ('forward' in step):
            if (self.ind == 0) or (first == True):
                if (self.ind == 0) and (first == True):
                    orig_feature = out
                out = self.ins1_bias(self.ins1(self.conv1(out)))
                out = self.relu1(out)
            if (self.ind <= 1) or (first == True):    
                if (self.ind == 1) and (first == True):
                    orig_feature = out
                out = self.ins2_bias(self.ins2(self.conv2(out)))
                out = self.relu2(out)
            if (self.ind == 2) and (first == True):
                orig_feature = out
            out = self.ins3_bias(self.ins3(self.conv3(out)))
            out = self.relu3(out)
            block1 = out
            out = self.maxpool3(out)
            out = self.ins4_bias(self.ins4(self.conv4(out)))
            out = self.relu4(out)
            block2 = out
            out = self.maxpool4(out)
            out = self.flatten(out)
            out = self.fc1_bias(self.fc1(out))
            out = self.relu5(out)
            out = self.fc2_bias(self.fc2(out))
            out = self.relu6(out)
            out = self.fc3_bias(self.fc3(out))
        elif ('backward' in step):
            out = self.fc3(out, step='backward')
            out = self.relu6(out, step='backward')
            out = self.fc2(out, step='backward')
            out = self.relu5(out, step='backward')
            out = self.fc1(out, step='backward')
            out = self.flatten(out, step='backward')
            out = self.ins4(out) 
            out = self.maxpool4(out, step='backward')
            out = self.relu4(out, step='backward')
            block2_recon = out
            out = self.conv4(out, step='backward')
            out = self.ins3(out)
            out = self.maxpool3(out, step='backward')
            out = self.relu3(out, step='backward')
            block1_recon = out
            out = self.conv3(out, step='backward')  
            if(self.ind <= 1):
                out = self.ins2(out) 
                out = self.relu2(out, step='backward')
                out = self.conv2(out, step='backward')
            if (self.ind == 0):
                out = self.ins1(out) 
                out = self.relu1(out, step='backward')
                out = self.conv1(out, step='backward')
            
        if (inter==True) and ('forward' in step):
            return out, orig_feature, block1, block2
        elif (inter_recon==True) and ('backward' in step):
            return out, block1_recon, block2_recon
        else:
            return out

    def reset(self):
        """
        Resets the pooling and activation states
        """
        self.maxpool3.reset()
        self.maxpool4.reset()
        self.relu1.reset()
        self.relu2.reset()
        self.relu3.reset()
        self.relu4.reset()
        self.relu5.reset()
        self.relu6.reset()

    def run_cycles(self, data):
        # evaluate with all the iterations
        with torch.no_grad():
            data = data.cuda()
            self.reset()
            output, orig_feature, _, _ = self.forward(data, first=True, inter=True)
            ff_prev = orig_feature
            for i_cycle in range(self.cycles):
                reconstruct = self.forward(output, step='backward')
                ff_current = ff_prev + self.res_param * (reconstruct - ff_prev)
                output = self.forward(ff_current, first=False)
                ff_prev = ff_current

        return output
        
    def run_cycles_adv(self, data):
        data = data.cuda()
        self.reset()
        output, orig_feature, _, _ = self.forward(data, first=True, inter=True)
        ff_prev = orig_feature
        for i_cycle in range(self.cycles):
            reconstruct = self.forward(output, step='backward')
            ff_current = ff_prev + self.res_param * (reconstruct - ff_prev)
            output = self.forward(ff_current, first=False)
            ff_prev = ff_current
        return output

    def run_average(self, data):
        # return averaged logits
        data = data.cuda()
        self.reset()
        output_list = []
        output, orig_feature, _, _ = self.forward(data, first=True, inter=True)
        ff_prev = orig_feature
        output_list.append(output)
        totaloutput = torch.zeros(output.shape).cuda()
        for i_cycle in range(self.cycles):
            reconstruct = self.forward(output, step='backward')
            ff_current = ff_prev + self.res_param * (reconstruct - ff_prev)
            ff_prev = ff_current
            output = self.forward(ff_current, first=False)
            output_list.append(output)
        for i in range(len(output_list)):
            totaloutput += output_list[i]
        return totaloutput / (self.cycles+1)
    
    def forward_adv(self, data):
        # run the first forward pass
        data = data.cuda()
        self.reset()
        output = self.forward(data)
        return output




