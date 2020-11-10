from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from shutil import copyfile
from datetime import datetime
from cnnf.model_cifar import WideResNet
from cnnf.model_mnist import CNNF
from eval import Evaluator
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description='CNNF testing')
    parser.add_argument('--dataset', choices=['cifar10', 'fashion'],
                        default='cifar10', help='the dataset for training the model')
    parser.add_argument('--test', choices=['average', 'last'],
                        default='average', help='output averaged logits or logits from the last iteration')
    parser.add_argument('--csv-dir', default='results.csv',
                        help='Directory for Saving the Evaluation results')
    parser.add_argument('--model-dir', default='models',
                        help='Directory for Saved Models')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    clean_dir = 'data/'
    
    # load in data
    if args.dataset=='cifar10':
        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(clean_dir, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])),
            batch_size=64, shuffle=True,
            num_workers=4, pin_memory=True)
        eps = 0.063
        eps_iter = 0.02
        nb_iter = 7

    elif args.dataset == 'fashion':
        dataloader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(clean_dir, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=100, shuffle=True)
        eps = 0.2
        eps_iter = 0.071
        nb_iter = 7

    log_acc_path = args.csv_dir
    evalmethod = args.test
    model_dir = args.model_dir

    with open(log_acc_path, 'a') as f:
        f.write(',clean,pgd_first,pgd_last,spsa_first,spsa_last,transfer,')
        f.write('\n')

    # Attacker model
    if args.dataset=='cifar10':
        model1_name = 'CNN_cifar.pt'
        model1_path = os.path.join(model_dir, model1_name)
        model1 = WideResNet(40, 10, 2, 0.0, ind=0, cycles=0, res_param=0.0).to(device)
        model1.load_state_dict(torch.load(model1_path))

    elif args.dataset == 'fashion':
        model1_name = 'CNN_fmnist.pt'
        model1_path = os.path.join(model_dir, model1_name)
        model1 = CNNF(10, 0, 0, 0.0).to(device)
        model1.load_state_dict(torch.load(model1_path))


    # Model to evaluate
    if args.dataset=='cifar10':
        model_name = 'CNNF_2_cifar.pt'
        model = WideResNet(40, 10, 2, 0.0, ind=5, cycles=2, res_param=0.1).to(device)
    elif args.dataset == 'fashion':
        model_name = 'CNNF_1_fmnist.pt'
        model = CNNF(10, ind=2, cycles=1, res_param=0.1).to(device)    

    model_path = os.path.join(model_dir, model_name)
    model.load_state_dict(torch.load(model_path))
    eval = Evaluator(device, model)
    clean_acc = eval.clean_accuracy(dataloader, test=evalmethod)
    
    # adv attack
    pgd_acc_first = eval.attack_pgd(dataloader, test=evalmethod, epsilon=eps, eps_iter=eps_iter, ete=False, nb_iter=nb_iter)
    pgd_acc_ete = eval.attack_pgd(dataloader, test=evalmethod, epsilon=eps, eps_iter=eps_iter, ete=True, nb_iter=nb_iter)

    spsa_acc_first = eval.attack_spsa(dataloader, test=evalmethod, epsilon=eps, ete=False, nb_iter=nb_iter)
    spsa_acc_ete = eval.attack_spsa(dataloader, test=evalmethod, epsilon=eps, ete=True, nb_iter=nb_iter)

    transfer_acc = eval.attack_pgd_transfer(model1, dataloader, test=evalmethod, epsilon=eps, eps_iter=eps_iter, nb_iter=nb_iter)

    with open(log_acc_path, 'a') as f:
        f.write('%s,' % model_name)
        f.write('%0.2f,' % (100. * clean_acc))
        f.write('%0.2f,' % (100. * pgd_acc_first))
        f.write('%0.2f,' % (100. * pgd_acc_ete))
        f.write('%0.2f,' % (100. * spsa_acc_first))
        f.write('%0.2f,' % (100. * spsa_acc_ete))
        f.write('%0.2f,' % (100. * transfer_acc))
        f.write('\n')

if __name__ == '__main__':
    main()

