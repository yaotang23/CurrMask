#!/bin/bash

declare -a tasks=("jaco_reach_top_left")
declare -a ts=("300000")
declare -a mdp_times=("2024.04.20/050734_mdp")
declare -a mask_types=("random")
declare -a mask_lens=("1")
declare -a data_seeds=("1")

for k in "${!data_seeds[@]}"; do
    seed=${data_seeds[$k]}
    for i in "${!tasks[@]}"; do
        task=${tasks[$i]}
        for j in "${!mdp_times[@]}"; do
            mask_type=${mask_types[$j]}
            mask_len=${mask_lens[$j]}
            mdp_time=${mdp_times[$j]}
            for snapshot_ts in "${ts[@]}"; do
                CUDA_VISIBLE_DEVICES=0 python finetune_rl.py \
                    pretrained_data=sup \
                    finetuned_data=semi \
                    agent=mdp_rl \
                    agent.batch_size=256 \
                    task=jaco_reach_top_left \
                    data_seed=${seed}\
                    snapshot_base_dir=/home/your_entity/currmask/output/final_mt/${mdp_time}/snapshot \
                    replay_buffer_dir=/data/your_entity_hdd/currmask_data \
                    snapshot_ts=$snapshot_ts \
                    project=jaco_eval_rl \
                    mt=true\
                    use_wandb=True \
                    pretrained_mask_type=${mask_type} \
                    pretrained_mask_len=${mask_len} &
                sleep 60
                CUDA_VISIBLE_DEVICES=1 python eval_plan.py \
                    pretrained_data=sup \
                    agent=mdp_goal \
                    agent.batch_size=384 \
                    num_eval_episodes=100 \
                    task=${task} \
                    snapshot_base_dir=/home/your_entity/currmask/output/final_mt/${mdp_times[$j]}/snapshot \
                    replay_buffer_dir=/data/your_entity_hdd/currmask_data \
                    goal_buffer_dir=/data/your_entity_hdd/currmask_data \
                    snapshot_ts=$snapshot_ts \
                    project=final-eval-plan \
                    use_wandb=True \
                    pretrained_mask_type=${mask_type}\
                    pretrained_mask_len=${mask_len} \
                    data_seed=${seed} &
                sleep 60
                CUDA_VISIBLE_DEVICES=2 python eval_prompt.py \
                    pretrained_data=sup \
                    agent=mdp_prompt \
                    finetuned_data='val' \
                    agent.batch_size=128 \
                    agent.context_length=8 \
                    agent.forecast_length=120\
                    task=${task} \
                    snapshot_base_dir=/home/your_entity/currmask/output/final_mt/${mdp_times[$j]}/snapshot \
                    replay_buffer_dir=/data/your_entity_hdd/currmask_data \
                    goal_buffer_dir=/data/your_entity_hdd/currmask_data \
                    snapshot_ts=${snapshot_ts} \
                    num_eval_episodes=100 \
                    project=final-eval-prompt \
                    use_wandb=True \
                    pretrained_mask_type=${mask_types[$j]} \
                    pretrained_mask_len=${mask_lens[$j]} \
                    pretrained_mask_ratio=${mask_ratios[$j]} \
                    data_seed=${data_seed} \
                    goal_buffer_num_workers=4 \
                    save_video=true &
                sleep 60
            done
        done
    done
done