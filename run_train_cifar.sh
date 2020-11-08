#!/bin/bash

python train.py --data 'cifar10' \
                --max-cycles 2 \
                --ind 5 \
                --mse-parameter 0.1 \
                --res-parameter 0.1 \
                --clean 'supclean' \
                --clean-parameter 0.05 \
                --lr 0.05 \
                --batch-size 256 \
                --eps 0.063 \
                --eps-iter 0.02 \
                --schedule 'poly' \
                --epochs 500 \
                --seed 0 \
                --grad-clip \
                --save-model 'CNNF_2_cifar' \
                --model-dir 'models'


