import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
import logging
import os
import torch.optim as optim
import numpy as np
import math
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pdb
import shutil
from tensorboardX import SummaryWriter

class BasicBlock(nn.Module):
    """Basic ResNet block."""

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, res_param=0.1):
        super(BasicBlock, self).__init__()
        self.ins1 = layers.InsNorm()
        self.ins1_bias = layers.Bias((1,in_planes,1,1))
        self.relu1 = layers.resReLU(res_param)
        self.conv1 = layers.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.ins2 = layers.InsNorm()
        self.ins2_bias = layers.Bias((1,out_planes,1,1))
        self.relu2 = layers.resReLU(res_param)
        self.conv2 = layers.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.is_in_equal_out = (in_planes == out_planes)
        self.conv_shortcut = (not self.is_in_equal_out) and layers.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False) or None

    def forward(self, x, step='forward'):
        if ('forward' in step):
            if not self.is_in_equal_out:
                x = self.relu1(self.ins1_bias(self.ins1(x)))
            else:
                out = self.relu1(self.ins1_bias(self.ins1(x)))
            if self.is_in_equal_out:
                out = self.relu2(self.ins2_bias(self.ins2(self.conv1(out))))
            else:
                out = self.relu2(self.ins2_bias(self.ins2(self.conv1(x))))
            if self.drop_rate > 0:
                out = layers.Dropout(out, p=self.drop_rate, training=self.training)
            out = self.conv2(out)
            if not self.is_in_equal_out:
                return torch.add(self.conv_shortcut(x), out)
            else:
                return torch.add(x, out)

        elif ('backward' in step):
            out = self.ins2(self.conv2(x, step='backward'))
            if self.drop_rate > 0:
                out = layers.Dropout(out, p=self.drop_rate, training=self.training, step='backward')
            out = self.relu2(out, step='backward')
            out = self.ins1(self.conv1(out, step='backward'))
            if not self.is_in_equal_out:
                out = torch.add(self.conv_shortcut(x, step='backward'), out)
            out = self.relu1(out, step='backward')
            if self.is_in_equal_out:
                out = torch.add(x, out)
            return out

class NetworkBlock(nn.Module):
    """Layer container for blocks."""
    def __init__(self,
               nb_layers,
               in_planes,
               out_planes,
               block,
               stride,
               drop_rate=0.0,
               ind=0,
               res_param=0.1):
        super(NetworkBlock, self).__init__()
        self.nb_layers = nb_layers
        self.res_param = res_param
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                      stride, drop_rate)
        # index of basic block to reconstruct to. 
        self.ind = ind

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                  drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(
              block(i == 0 and in_planes or out_planes, out_planes,
                    i == 0 and stride or 1, drop_rate, self.res_param))
        return nn.ModuleList(layers)

    def forward(self, x, step='forward', first=True, inter=False):
        # first: the first forward pass is the same as conventional CNN.
        # inter: if True, return intemediate features. 
        
        # reconstruct to pixel level
        if (self.ind==0):
            if ('forward' in step):
                for block in self.layer:
                    x = block(x)
            elif ('backward' in step):
                for block in self.layer[::-1]:
                    x = block(x, step='backward')

        # reconstruct to intermediate layers
        elif (self.ind>0):        
            if ('forward' in step):
                if (first==True):
                    if(inter==False):
                        for block in self.layer:
                            x = block(x)
                    elif(inter==True):
                        for idx, block in enumerate(self.layer):
                            x = block(x)
                            if ((idx+1)==self.ind):
                                orig_feature = x        
                elif (first==False):
                    for idx, block in enumerate(self.layer):
                        if (idx+1 > self.ind):
                            x = block(x)  
            elif ('backward' in step):
                ind_back = self.nb_layers-self.ind
                for idx, block in enumerate(self.layer[::-1]):
                    if (idx < ind_back):
                        x = block(x, step='backward')    
        if (inter==False):
            return x
        elif (inter==True):
            return x, orig_feature


class WideResNet(nn.Module):
    """ CNNF on Wide ResNet Architecture. """

    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0, ind=0, cycles=2, res_param=0.1):
        super(WideResNet, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock
        self.ind = ind
        self.res_param = res_param
        self.cycles = cycles
        # 1st conv before any network block
        self.conv1 = layers.Conv2d(
            3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1,
                                   drop_rate, self.ind, res_param=self.res_param)
        # 2nd block
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2,
                                   drop_rate, res_param=self.res_param)
        # 3rd block
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2,
                                   drop_rate, res_param=self.res_param)
        # global average pooling and classifier
        self.ins1 = layers.InsNorm()
        self.ins1_bias = layers.Bias((1,n_channels[3],1,1))
        self.relu = layers.resReLU(res_param)
        self.fc = layers.Linear(n_channels[3], num_classes)
        self.fc_bias = layers.Bias((1,num_classes))
        self.flatten = layers.Flatten()
        self.n_channels = n_channels[3]
        self.ins0 = layers.InsNorm()

    def forward(self, out, step='forward', first=True, inter=False, inter_recon=False):     
        if ('forward' in step):
            if (self.ind==0):
                if(inter==True):
                    orig_feature = out
                out = self.conv1(out)
                out = self.block1(out, first=first, inter=False)

            elif (self.ind>0):
                if(first==True):
                    out = self.conv1(out)
                if (inter==True):
                    out, orig_feature = self.block1(out, first=first, inter=inter)
                elif (inter==False):
                    out = self.block1(out, first=first, inter=inter)

            if (inter==True):
                block1 = out
            out = self.block2(out)
            if (inter==True):
                block2 = out
            out = self.block3(out)
            if (inter==True):
                block3 = out
            out = self.relu(self.ins1_bias(self.ins1(out)))
            out = F.avg_pool2d(out, 8)
            out = self.flatten(out)
            out = self.fc_bias(self.fc(out))

        elif ('backward' in step):
            out = self.fc(out, step='backward')
            out = self.flatten(out, step='backward')
            out = F.interpolate(out, scale_factor=8)
            out = self.ins1(out)
            out = self.relu(out, step='backward')
            block3_recon = out
            out = self.block3(out, step='backward')
            block2_recon = out
            out = self.block2(out, step='backward')
            block1_recon = out
            out = self.block1(out, step='backward', first=first, inter=inter)
            if (self.ind==0):
                out = self.conv1(out, step='backward')
            
        if (inter==True) and ('forward' in step):
            return out, orig_feature, block1, block2, block3
        elif (inter_recon==True) and ('backward' in step):
            return out, block1_recon, block2_recon, block3_recon
        else:
            return out

    def reset(self):
        """
        Resets the pooling and activation states
        """
        self.relu.reset()
        
        for BasicBlock in self.block1.layer:
            BasicBlock.relu1.reset()
            BasicBlock.relu2.reset()

        for BasicBlock in self.block2.layer:
            BasicBlock.relu1.reset()
            BasicBlock.relu2.reset()
            
        for BasicBlock in self.block3.layer:
            BasicBlock.relu1.reset()
            BasicBlock.relu2.reset()

    def run_cycles(self, data):
        # evaluate with all the iterations
        with torch.no_grad():
            data = data.cuda()
            self.reset()
            output, orig_feature, _, _, _ = self.forward(data, first=True, inter=True)
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
        output, orig_feature, _, _, _ = self.forward(data, first=True, inter=True)
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
        output, orig_feature, _, _, _ = self.forward(data, first=True, inter=True)
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
