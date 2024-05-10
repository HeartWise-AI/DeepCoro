#! /bin/sh

## Call when running from dockerfile
# python make_batch_inference.py \
#     --input_path /dcm_input/input_file.csv \
#     --save_dir /results/ \
#     --params_file /params.json


## Call when working local 
# python make_batch_inference.py \
#    --input_path /volume/deepcoro/repotest/DeepCoro/random_dicoms/input_file.csv \
#    --save_dir /volume/deepcoro/repotest/DeepCoro/results/test/batch_inference/ \
#    --params_file /volume/deepcoro/repotest/DeepCoro/params.json