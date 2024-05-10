#! /bin/sh

## Call when running from dockerfile
python make_batch_inference.py \
    --workdir /

## Call when working local 
# python make_batch_inference.py \
#    --input_path /volume/deepcoro/repotest/DeepCoro/random_dicoms/input_file.csv \
#    --save_dir /volume/deepcoro/repotest/DeepCoro/results/test/batch_inference/ \
#    --params_file /volume/deepcoro/repotest/DeepCoro/params.json