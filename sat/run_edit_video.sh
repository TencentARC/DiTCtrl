#! /bin/bash

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"



inference_case_config="inference_case_configs/edit/suv.yaml"
run_cmd="$environs python sample_video_edit.py --base configs/cogvideox_2b.yaml configs/inference.yaml --custom-config $inference_case_config"
echo ${run_cmd}
eval ${run_cmd}


inference_case_config="inference_case_configs/edit/fox_wolf.yaml"
run_cmd="$environs python sample_video_edit.py --base configs/cogvideox_2b.yaml configs/inference.yaml --custom-config $inference_case_config"
echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
