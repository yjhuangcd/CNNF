import torch
import torch.nn as nn
import torch.nn.functional as F
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

from advertorch.attacks import GradientSignAttack, LinfPGDAttack, LinfSPSAAttack
from advertorch.context import ctx_noparamgrad_and_eval

class Evaluator:
    def __init__(self, device, model):
        self.device = device
        self.model = model

    def clean_accuracy(self, clean_loader, test='last'):
        """ Evaluate the model on clean dataset. """
        self.model.eval()
        
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(clean_loader):
                data, target = data.to(self.device), target.to(self.device) 
                if (test=='last'):
                    output = self.model.run_cycles(data)
                elif(test=='average'):
                    output = self.model.run_average(data)
                else:
                    self.model.reset()
                    output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True) 
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / len(clean_loader.dataset)
        print('Clean Test Acc {:.3f}'.format(100. * acc))

        return acc
    
    def attack_pgd(self, clean_loader, epsilon=0.1, eps_iter=0.02, test='average', ete=False, nb_iter=7):
        """ Use PGD to attack the model. """
        
        self.model.eval()
        self.model.reset()

        if (ete==False):
            adv_func = self.model.forward_adv
        else:
            adv_func = self.model.run_cycles_adv  
                
        adversary = LinfPGDAttack(
            adv_func, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon, 
            nb_iter=nb_iter, eps_iter=eps_iter, rand_init=True, clip_min=-1.0, clip_max=1.0, targeted=False)
                
        correct = 0
        for batch_idx, (data, target) in enumerate(clean_loader):
            data, target = data.to(self.device), target.to(self.device)     
            self.model.reset()
            with ctx_noparamgrad_and_eval(self.model):
                adv_images = adversary.perturb(data, target)

            if(test=='last'):
                output = self.model.run_cycles(adv_images)
            elif(test=='average'):
                output = self.model.run_average(adv_images)
            else:
                self.model.reset()
                output = self.model(adv_images)
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / len(clean_loader.dataset)
        print('PGD attack Acc {:.3f}'.format(100. * acc))

        return acc
    
    def attack_spsa(self, clean_loader, epsilon=0.1, test='average', ete=False, nb_iter=7):
        """ Use SPSA to attack the model. """
        
        self.model.eval()
        self.model.reset()
        if (ete==False):
            adv_func = self.model.forward_adv
        else:
            adv_func = self.model.run_cycles_adv    
        
        adversary = LinfSPSAAttack(
            adv_func, loss_fn=nn.CrossEntropyLoss(reduction="none"), eps=epsilon, 
            nb_iter=nb_iter, delta=0.01, nb_sample=128, max_batch_size=64, clip_min=-1.0, clip_max=1.0, targeted=False)
            
        correct = 0
        numofdata = 0
        for batch_idx, (data, target) in enumerate(clean_loader):
            # To speed up the evaluation of attack, evaluate the first 10 batches
            if(batch_idx < 10): 
                data, target = data.to(self.device), target.to(self.device)     
                self.model.reset()
                with ctx_noparamgrad_and_eval(self.model):
                    adv_images = adversary.perturb(data, target)

                if(test=='last'):
                    output = self.model.run_cycles(adv_images)
                elif(test=='average'):
                    output = self.model.run_average(adv_images)
                else:
                    self.model.reset()
                    output = self.model(adv_images)
                pred = output.argmax(dim=1, keepdim=True) 
                correct += pred.eq(target.view_as(pred)).sum().item()
                numofdata += data.shape[0]

        acc = correct / numofdata
        print('SPSA attack Acc {:.3f}'.format(100. * acc))

        return acc
    
    def attack_pgd_transfer(self, model_attacker, clean_loader, epsilon=0.1, eps_iter=0.02, test='average', nb_iter=7):
        """ Use adversarial samples generated against model_attacker to attack the current model. """
        
        self.model.eval()
        self.model.reset()
        model_attacker.eval()
        model_attacker.reset()
        adversary = LinfPGDAttack(
            model_attacker.forward_adv, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon, 
            nb_iter=nb_iter, eps_iter=eps_iter, rand_init=True, clip_min=-1.0, clip_max=1.0, targeted=False)
        
        correct = 0
        for batch_idx, (data, target) in enumerate(clean_loader):
            data, target = data.to(self.device), target.to(self.device)            
            self.model.reset()
            model_attacker.reset()
            with ctx_noparamgrad_and_eval(model_attacker):
                adv_images = adversary.perturb(data, target)

                if(test=='last'):
                    output = self.model.run_cycles(adv_images)
                elif(test=='average'):
                    output = self.model.run_average(adv_images)
                else:
                    self.model.reset()
                    output = self.model(adv_images)
                    
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / len(clean_loader.dataset)
        print('PGD attack Acc {:.3f}'.format(100. * acc))

        return acc
