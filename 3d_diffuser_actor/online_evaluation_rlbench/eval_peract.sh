depth_model=unidepthfinetune
exp=3d_diffuser_actor_${depth_model}
exp=3d_diffuser_actor_${depth_model}_4views

tasks=(
    meat_off_grill
    # slide_block_to_color_target open_drawer sweep_to_dustpan_of_size meat_off_grill put_item_in_drawer
    # open_drawer slide_block_to_color_target sweep_to_dustpan_of_size meat_off_grill put_item_in_drawer
    # close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap place_cups
    # place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap place_cups
)
# data_dir=./data/peract/raw/test/
data_dir=/project2/yehhh/datasets/RLBench/raw/test
num_episodes=100
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
use_instruction=1
max_tries=2
verbose=1
interpolation_length=2
single_task_gripper_loc_bounds=0
embedding_dim=120
cameras="left_shoulder,right_shoulder,wrist,front"
# cameras="left_shoulder,right_shoulder,front"
fps_subsampling_factor=5
lang_enhanced=0
relative_action=0
seed=2
checkpoint=train_logs/diffuser_actor_peract.pth
# checkpoint=/project/yi-ray/3d_diffuser_actor/train_logs/open_drawer_wrist/diffusion_multitask-C120-B8-lr1e-4-DI1-2-H3-DT100/best.pth
quaternion_format=wxyz  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint
# cp ../RLBench/rlbench/backend/scene.py /project/yehhh/UIUC/programs/RLBench/rlbench/backend/
# cp ../RLBench/rlbench/action_modes/action_mode.py /project/yehhh/UIUC/programs/RLBench/rlbench/action_modes/
# cp ../RLBench/rlbench/task_environment.py  /project/yehhh/UIUC/programs/RLBench/rlbench/
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PYTHONPATH:/data1/yehhh_/RL_2025_Spring_Final/UniDepth

num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
    # CUDA_LAUNCH_BLOCKING=1 python online_evaluation_rlbench/evaluate_policy.py \
    python online_evaluation_rlbench/evaluate_policy.py \
    --tasks ${tasks[$i]} \
    --checkpoint $checkpoint \
    --diffusion_timesteps 100 \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --relative_action $relative_action \
    --num_history 3 \
    --test_model 3d_diffuser_actor \
    --cameras $cameras \
    --verbose $verbose \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 1 \
    --embedding_dim $embedding_dim \
    --rotation_parametrization "6D" \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/$exp/seed$seed/${tasks[$i]}.json  \
    --use_instruction $use_instruction \
    --instructions instructions/peract/instructions.pkl \
    --variations {0..60} \
    --max_tries $max_tries \
    --max_steps 25 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.04 \
    --quaternion_format $quaternion_format \
    --interpolation_length $interpolation_length \
    --dense_interpolation 1 \
    --use_mono_depth 1 \
    --mono_depth_model_name $depth_model 
done

