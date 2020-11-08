#!/bin/bash

python train.py --data 'fashion' \
                --max-cycles 1 \
                --ind 2 \
                --mse-parameter 0.1 \
                --res-parameter 0.1 \
                --clean 'supclean' \
                --clean-parameter 0.1 \
                --lr 0.05 \
                --batch-size 256 \
                --eps 0.2 \
                --eps-iter 0.071 \
                --schedule 'poly' \
                --epochs 200 \
                --seed 1 \
                --grad-clip \
                --save-model 'CNNF_1_fmnist' \
                --model-dir 'models'


wait 
echo "All done"
