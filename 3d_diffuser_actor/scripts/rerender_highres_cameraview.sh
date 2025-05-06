#!/usr/bin/bash

SPLIT=train
DEMO_ROOT=/project2/yehhh/datasets/RLBench/raw/${SPLIT}
RAW_SAVE_PATH=/project2/yehhh/datasets/RLBench/raw_highres/${SPLIT}
PACKAGE_SAVE_PATH=/project2/yehhh/datasets/RLBench/packaged_highres/${SPLIT}
depth_model=unidepthfinetune

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd):$PYTHONPATH
# export PYTHONPATH=$PYTHONPATH:/data1/yehhh_/RL_2025_Spring_Final/UniDepth

# Re-render high-resolution camera views
# for task in place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap
# for task in open_drawer sweep_to_dustpan_of_size meat_off_grill put_item_in_drawer slide_block_to_color_target
for task in meat_off_grill 
do
    python data_preprocessing/rerender_highres_rlbench.py \
        --tasks=$task \
        --save_path=$RAW_SAVE_PATH \
        --demo_path=$DEMO_ROOT \
        --image_size=256,256\
        --renderer=opengl \
        --processes=1 \
        --all_variations=True \
        --use_mono_depth 1 \
        --mono_depth_model_name $depth_model 

done

# Re-arrange directory
python data_preprocessing/rearrange_rlbench_demos.py --root_dir /project2/yehhh/datasets/RLBench/raw_highres/${SPLIT}

# Package episodes into .dat fiels
# for task in place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap
for task in slide_block_to_color_target open_drawer sweep_to_dustpan_of_size meat_off_grill put_item_in_drawer
do
    python data_preprocessing/package_rlbench.py \
        --data_dir=$RAW_SAVE_PATH \
        --tasks=$task \
        --output=$PACKAGE_SAVE_PATH \
        --store_intermediate_actions=1
done

# for task in slide_block_to_color_target open_drawer sweep_to_dustpan_of_size meat_off_grill put_item_in_drawer
# do
#     unzip ${task}.zip 
# done