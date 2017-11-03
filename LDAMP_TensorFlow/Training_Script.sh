#!/usr/bin/env bash
#Training all these networks will take a few days on a Titan X GPU

python ./TrainLearnedDAMP.py --alg="DAMP" --init_method="smaller_net" --start_layer=1 --DnCNN_layer=16

python ./TrainLearnedDAMP.py --alg="DAMP" --init_method="layer_by_layer" --start_layer=10  --train_end_to_end --DnCNN_layer=16

python ./TrainLearnedDAMP.py --alg="DIT" --init_method="smaller_net" --start_layer=1 --DnCNN_layer=16

python ./TrainLearnedDAMP.py --alg="DIT" --init_method="layer_by_layer" --start_layer=10  --train_end_to_end --DnCNN_layer=16

python ./TrainDnCNN.py --sigma_w_min=00. --sigma_w_max=10.

python ./TrainDnCNN.py --sigma_w_min=10. --sigma_w_max=20.

python ./TrainDnCNN.py --sigma_w_min=20. --sigma_w_max=40.

python ./TrainDnCNN.py --sigma_w_min=40. --sigma_w_max=60.

python ./TrainDnCNN.py --sigma_w_min=60. --sigma_w_max=80.

python ./TrainDnCNN.py --sigma_w_min=80. --sigma_w_max=100.

python ./TrainDnCNN.py --sigma_w_min=100. --sigma_w_max=150.

python ./TrainDnCNN.py --sigma_w_min=150. --sigma_w_max=300.

python ./TrainDnCNN.py --sigma_w_min=300. --sigma_w_max=500.
