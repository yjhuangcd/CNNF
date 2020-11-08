from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from shutil import copyfile
from datetime import datetime
import os
import logging
import numpy as np
import random
import math
import torchvision
from torchvision import datasets
from torchvision import transforms
import pdb
import shutil
from tensorboardX import SummaryWriter
from model_cifar import WideResNet
from model_mnist import CNNF

from utils import *
from advertorch.attacks import GradientSignAttack, LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval

def train_adv(args, model, device, train_loader, optimizer, scheduler, epoch,
          cycles, mse_parameter=1.0, clean_parameter=1.0, clean='supclean'):

    model.train()

    correct = 0
    train_loss = 0.0

    model.reset()

    adversary = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.eps, 
        nb_iter=args.nb_iter, eps_iter=args.eps_iter, rand_init=True, clip_min=-1.0, clip_max=1.0, targeted=False)
 
    for batch_idx, (images, targets) in enumerate(train_loader):
            
        optimizer.zero_grad()
        images = images.cuda()
        targets = targets.cuda()

        model.reset()
        with ctx_noparamgrad_and_eval(model):
            adv_images = adversary.perturb(images, targets)

        images_all = torch.cat((images, adv_images), 0)
              
        # Reset the model latent variables
        model.reset() 
        if (args.dataset == 'cifar10'):
            logits, orig_feature_all, block1_all, block2_all, block3_all = model(images_all, first=True, inter=True)
        elif (args.dataset == 'fashion'):
            logits, orig_feature_all, block1_all, block2_all = model(images_all, first=True, inter=True)
        ff_prev = orig_feature_all
        # find the original feature of clean images
        orig_feature, _ = torch.split(orig_feature_all, images.size(0))
        block1_clean, _ = torch.split(block1_all, images.size(0))
        block2_clean, _ = torch.split(block2_all, images.size(0))
        if (args.dataset == 'cifar10'):
            block3_clean, _ = torch.split(block3_all, images.size(0))
        logits_clean, logits_adv = torch.split(logits, images.size(0))
        
        if not ('no' in clean):
            loss = (clean_parameter * F.cross_entropy(logits_clean, targets) + F.cross_entropy(logits_adv, targets)) / (2*(cycles+1))
        else:        
            loss = F.cross_entropy(logits_adv, targets) / (cycles+1) 

        for i_cycle in range(cycles):
            if (args.dataset == 'cifar10'):
                recon, block1_recon, block2_recon, block3_recon = model(logits, step='backward', inter_recon=True)
            elif (args.dataset == 'fashion'):    
                recon, block1_recon, block2_recon = model(logits, step='backward', inter_recon=True)
            recon_clean, recon_adv = torch.split(recon, images.size(0))
            recon_block1_clean, recon_block1_adv = torch.split(block1_recon, images.size(0))
            recon_block2_clean, recon_block2_adv = torch.split(block2_recon, images.size(0))
            if (args.dataset == 'cifar10'):
                recon_block3_clean, recon_block3_adv = torch.split(block3_recon, images.size(0))
                loss += (F.mse_loss(recon_adv, orig_feature) + F.mse_loss(recon_block1_adv, block1_clean) + F.mse_loss(recon_block2_adv, block2_clean) + F.mse_loss(recon_block3_adv, block3_clean)) * mse_parameter / (4*cycles)
            elif (args.dataset == 'fashion'):
                loss += (F.mse_loss(recon_adv, orig_feature) + F.mse_loss(recon_block1_adv, block1_clean) + F.mse_loss(recon_block2_adv, block2_clean)) * mse_parameter / (3*cycles)

            # feedforward    
            ff_current = ff_prev + args.res_parameter * (recon - ff_prev)
            logits = model(ff_current, first=False)
            ff_prev = ff_current
            logits_clean, logits_adv = torch.split(logits, images.size(0)) 
            if not ('no' in clean):
                loss += (clean_parameter * F.cross_entropy(logits_clean, targets) + F.cross_entropy(logits_adv, targets)) / (2*(cycles+1))
            else:
                loss += F.cross_entropy(logits_adv, targets) / (cycles+1) 
            
        pred = logits_clean.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(targets.view_as(pred)).sum().item()

        loss.backward()
        if (args.grad_clip):
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        train_loss += loss
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader)
    acc = correct / len(train_loader.dataset)
    return train_loss, acc

def test(args, model, device, test_loader, cycles, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    
    noise_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # Calculate accuracy with the original images
            model.reset()
            if (args.dataset == 'cifar10'):
                output, orig_feature, _, _, _ = model(data, first=True, inter=True)
            else:
                output, orig_feature, _, _ = model(data, first=True, inter=True)
            ff_prev = orig_feature
            for i_cycle in range(cycles):
                recon = model(output, step='backward')
                ff_current = ff_prev + args.res_parameter * (recon - ff_prev)
                output = model(ff_current, first=False)

            test_loss += F.cross_entropy(output, target, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, correct / len(test_loader.dataset)

def test_pgd(args, model, device, test_loader, epsilon=0.063):
    
    model.eval()
    model.reset()        
    adversary = LinfPGDAttack(
        model.forward_adv, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon, 
        nb_iter=args.nb_iter, eps_iter=args.eps_iter, rand_init=True, clip_min=-1.0, clip_max=1.0, targeted=False)

    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        model.reset()
        with ctx_noparamgrad_and_eval(model):
            adv_images = adversary.perturb(data, target)

        output = model.run_cycles(adv_images)

        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()

    acc = correct / len(test_loader.dataset)
    print('PGD attack Acc {:.3f}'.format(100. * acc))

    return acc

def main():
    parser = argparse.ArgumentParser(description='CNNF training')
    # optimization parameters
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128 for CIFAR, 64 for MNIST)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.15, metavar='LR',
                        help='learning rate (default: 0.05 for SGD)')
    parser.add_argument('--power', type=float, default=0.9, metavar='LR',
                        help='learning rate for poly scheduling')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
    parser.add_argument('--grad-clip', action='store_true', default=False,
                        help='enable gradient clipping')
    parser.add_argument('--dataset', choices=['cifar10', 'fashion'],
                        default='cifar10', help='the dataset for training the model')
    parser.add_argument('--schedule', choices=['poly', 'cos'],
                        default='poly', help='scheduling for learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=400, metavar='N',
                        help='how many batches to wait before logging training status')
    
    # adversarial training parameters
    parser.add_argument('--eps', type=float, default=0.063,
                        help='Perturbation magnitude for adv training')
    parser.add_argument('--eps-iter', type=float, default=0.02,
                        help='attack step size')
    parser.add_argument('--nb_iter', type=int, default=7,
                        help='number of steps in pgd attack')
    parser.add_argument('--clean', choices=['no', 'supclean'],
                        default='supclean', help='whether to use clean data in adv training')
    
    # hyper-parameters
    parser.add_argument('--mse-parameter', type=float, default=1.0,
                        help='weight of the reconstruction loss')
    parser.add_argument('--clean-parameter', type=float, default=1.0,
                        help='weight of the clean Xentropy loss')
    parser.add_argument('--res-parameter', type=float, default=0.1,
                        help='step size for residuals')
    
    # model parameters
    parser.add_argument('--layers', default=40, type=int, help='total number of layers for WRN')
    parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor for WRN')
    parser.add_argument('--droprate', default=0.0, type=float, help='Dropout probability')
    parser.add_argument('--ind', type=int, default=2,
                        help='index of the intermediate layer to reconstruct to')
    parser.add_argument('--max-cycles', type=int, default=2,
                        help='the maximum cycles that the CNN-F uses')
    parser.add_argument('--save-model', default=None,
                        help='Name for Saving the current Model')
    parser.add_argument('--model-dir', default=None,
                        help='Directory for Saving the current Model')

    
    args = parser.parse_args()
 
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    seed_torch(args.seed) 
    
    Tensor_writer = SummaryWriter(os.path.join(args.model_dir, args.save_model))

    train_transform_cifar = transforms.Compose(
      [transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4),
       transforms.ToTensor(),
       transforms.Normalize([0.5] * 3, [0.5] * 3)])      
    
    test_transform_cifar = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize([0.5] * 3, [0.5] * 3)])
    
    transform_mnist = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))])  

    # Load datasets and architecture                   
    if args.dataset == 'fashion':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=True, download=True,
                           transform=transform_mnist),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=False, transform=transform_mnist),
            batch_size=args.test_batch_size, shuffle=True, drop_last=True)
        num_classes = 10
        model = CNNF(num_classes, ind=args.ind, cycles=args.max_cycles, res_param=args.res_parameter).to(device)
        
    elif args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(
            'data', train=True, transform=train_transform_cifar, download=True)
        test_data = datasets.CIFAR10(
            'data', train=False, transform=test_transform_cifar, download=True)
        train_loader = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
          test_data, batch_size=args.test_batch_size,
          shuffle=True, num_workers=4, pin_memory=True)
        num_classes = 10
        model = WideResNet(args.layers, 10, args.widen_factor, args.droprate, args.ind, args.max_cycles, args.res_parameter).to(device)
    
    optimizer = torch.optim.SGD(
          model.parameters(),
          args.lr,
          momentum=args.momentum,
          weight_decay=args.wd)
            
    if(args.schedule == 'cos'):        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
          optimizer, lr_lambda=lambda step: get_lr(step, args.epochs * len(train_loader), 1.0, 1e-5))
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
          optimizer, lr_lambda=lambda step: lr_poly(1.0, step, args.epochs * len(train_loader), args.power))

    # Begin training
    best_acc = 0

    for epoch in range(args.epochs):    
        train_loss, train_acc = train_adv(args, model, device, train_loader, optimizer, scheduler, epoch, 
          cycles=args.max_cycles, mse_parameter=args.mse_parameter, clean_parameter=args.clean_parameter, clean=args.clean)

        test_loss, test_acc = test(args, model, device, test_loader, cycles=args.max_cycles, epoch=epoch)
        
        Tensor_writer.add_scalars('loss', {'train': train_loss}, epoch)
        Tensor_writer.add_scalars('acc', {'train': train_acc}, epoch)

        Tensor_writer.add_scalars('loss', {'test': test_loss}, epoch)
        Tensor_writer.add_scalars('acc', {'test': test_acc}, epoch)

        # Save the model with the best accuracy
        if test_acc > best_acc and args.save_model is not None:
            best_acc = test_acc
            experiment_fn = args.save_model
            torch.save(model.state_dict(),
                       args.model_dir + "/{}-best.pt".format(experiment_fn))
                        
        if ((epoch+1)%50)==0 and args.save_model is not None:
            experiment_fn = args.save_model
            torch.save(model.state_dict(),
                       args.model_dir + "/{}-epoch{}.pt".format(experiment_fn,epoch))   
            pgd_acc = test_pgd(args, model, device, test_loader, epsilon=args.eps)

            Tensor_writer.add_scalars('pgd_acc', {'test': pgd_acc}, epoch)

    # Save final model
    if args.save_model is not None:
        experiment_fn = args.save_model
        torch.save(model.state_dict(),
                   args.model_dir + "/{}.pt".format(experiment_fn))


if __name__ == '__main__':
    main()



