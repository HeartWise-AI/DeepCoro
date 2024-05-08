#! /bin/sh

python make_batch_inference.py \
    --input_file_path /volume/deepcoro/repotest/DeepCoro/random_dicoms/input_file.csv \
    --models_dir /volume/deepcoro/repotest/DeepCoro/models/ \
    --save_dir /volume/deepcoro/repotest/DeepCoro/results/4_stenosis/batch_inference/ \ 
    --params_file /volume/deepcoro/repotest/DeepCoro/params.json \