#!/bin/bash
read -p "Cuda: " cuda
read -p "dataset: " dataset
read -p "outputdir: " savedir
read -p "n_layers: " layers
read -p "p_stages: " stages

CUDA_VISIBLE_DEVICES=$cuda python main.py --search --tune --tune_iter 10 --run_iter 1 --seed 69 --p_stages $stages  \
--increase_layers 2  --target_n_layers $layers  --cfg "configurations/$dataset.yaml" SAVE_DIR "$savedir/$dataset/$stages/$layers"
