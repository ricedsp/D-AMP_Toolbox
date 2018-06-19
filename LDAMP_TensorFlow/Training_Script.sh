#!/usr/bin/env bash
#Training all these networks will take a few days on Titan X GPUs

#GPU 0
{ \
CUDA_VISIBLE_DEVICES=0 python ./TrainDnCNN.py --sigma_w_min=00. --sigma_w_max=10.; \
CUDA_VISIBLE_DEVICES=0 python ./TrainDnCNN.py --sigma_w_min=10. --sigma_w_max=20.; \
CUDA_VISIBLE_DEVICES=0 python ./TrainDnCNN.py --sigma_w_min=20. --sigma_w_max=40.; \
CUDA_VISIBLE_DEVICES=0 python ./TrainDnCNN.py --sigma_w_min=40. --sigma_w_max=60.; \
CUDA_VISIBLE_DEVICES=0 python ./TrainDnCNN.py --sigma_w_min=60. --sigma_w_max=80.; \
CUDA_VISIBLE_DEVICES=0 python ./TrainDnCNN.py --sigma_w_min=80. --sigma_w_max=100.; }&

#GPU 1
{ \
CUDA_VISIBLE_DEVICES=1 python ./TrainDnCNN.py --sigma_w_min=100. --sigma_w_max=150.; \
CUDA_VISIBLE_DEVICES=1 python ./TrainDnCNN.py --sigma_w_min=150. --sigma_w_max=300.; \
CUDA_VISIBLE_DEVICES=1 python ./TrainDnCNN.py --sigma_w_min=300. --sigma_w_max=500.; \
CUDA_VISIBLE_DEVICES=1 python ./TrainDnCNN.py --sigma_w_min=10. --sigma_w_max=10.; \
CUDA_VISIBLE_DEVICES=1 python ./TrainDnCNN.py --sigma_w_min=20. --sigma_w_max=20.; }&

#GPU 2
{ \
CUDA_VISIBLE_DEVICES=2 python ./TrainDnCNN.py --sigma_w_min=25. --sigma_w_max=25.; \
CUDA_VISIBLE_DEVICES=2 python ./TrainDnCNN.py --sigma_w_min=40. --sigma_w_max=40.; \
CUDA_VISIBLE_DEVICES=2 python ./TrainDnCNN.py --sigma_w_min=60. --sigma_w_max=60.; \
CUDA_VISIBLE_DEVICES=2 python ./TrainDnCNN.py --sigma_w_min=80. --sigma_w_max=80.; \
CUDA_VISIBLE_DEVICES=2 python ./TrainDnCNN.py --sigma_w_min=100. --sigma_w_max=100.; }&


#GPU 3
{ \
CUDA_VISIBLE_DEVICES=3 python ./TrainLearnedDAMP.py --alg="DAMP" --init_method="smaller_net" --start_layer=1 --DnCNN_layer=16; \
CUDA_VISIBLE_DEVICES=3 python ./TrainLearnedDAMP.py --alg="DAMP" --init_method="layer_by_layer" --start_layer=10  --train_end_to_end --DnCNN_layer=16; }&


#GPU 4
{ \
CUDA_VISIBLE_DEVICES=4 python ./TrainLearnedDAMP.py --alg="DAMP" --init_method="smaller_net" --start_layer=1 --DnCNN_layer=16 --loss_func='SURE'; \
}&

#GPU 5
{ \
CUDA_VISIBLE_DEVICES=5 python ./TrainLearnedDAMP.py --alg="DAMP" --init_method="smaller_net" --start_layer=1 --DnCNN_layer=16 --loss_func='GSURE'; \
}&

#GPU 6
{ \
CUDA_VISIBLE_DEVICES=6 python ./TrainLearnedDAMP.py --alg="DIT" --init_method="smaller_net" --start_layer=1 --DnCNN_layer=16; \
CUDA_VISIBLE_DEVICES=6 python ./TrainLearnedDAMP.py --alg="DIT" --init_method="layer_by_layer" --start_layer=10  --train_end_to_end --DnCNN_layer=16; }&

#GPU 7
{ \
CUDA_VISIBLE_DEVICES=7 python ./TrainDnCNN.py --sigma_w_min=10. --sigma_w_max=10. --loss_func='SURE'; \
CUDA_VISIBLE_DEVICES=7 python ./TrainDnCNN.py --sigma_w_min=20. --sigma_w_max=20. --loss_func='SURE'; \
CUDA_VISIBLE_DEVICES=7 python ./TrainDnCNN.py --sigma_w_min=25. --sigma_w_max=25. --loss_func='SURE'; \
CUDA_VISIBLE_DEVICES=7 python ./TrainDnCNN.py --sigma_w_min=40. --sigma_w_max=40. --loss_func='SURE'; \
CUDA_VISIBLE_DEVICES=7 python ./TrainDnCNN.py --sigma_w_min=60. --sigma_w_max=60. --loss_func='SURE'; \
CUDA_VISIBLE_DEVICES=7 python ./TrainDnCNN.py --sigma_w_min=80. --sigma_w_max=80. --loss_func='SURE'; \
CUDA_VISIBLE_DEVICES=7 python ./TrainDnCNN.py --sigma_w_min=100. --sigma_w_max=100. --loss_func='SURE'; }&