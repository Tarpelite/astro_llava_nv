export CUDA_VISIBLE_DEVICES=0
python ft_qwen2_ultra.py \
    --model_path /mnt/data/CVPR2025/task1_data/Qwen2-VL-2B-Instruct \
    --train_regression_data /mnt/data/CVPR2025/task1_data/data1105/regression/train_no_classification_addfeat.hdf5 \
    --train_classification_data /mnt/data/CVPR2025/task1_data/data1109/final/train_processed.hdf5 \
    --regression_eval_dir /mnt/data/CVPR2025/task1_data/data1105/regression \
    --classification_eval_dir /mnt/data/CVPR2025/task1_data/data1109/final/classification_eval_split \
    --eval_regression_tasks task1_redshift task2_log_mstar task2_z_mw task2_tage_mw task2_sSFR\
    --eval_classification_tasks smooth disk-edge-on spiral-arms bar bulge-size how-rounded edge-on-bulge spiral-winding spiral-arm-count merging\
    --image_dir /mnt/data/CVPR2025/task1_data/images/images \
    --template_path template_ultra_qwen2vl_classification.json \
    --output_dir /mnt/data/CVPR2025/task1_data/ckpts/ft_qwen2VL_2B_ultra_1112 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_steps 1000 \
    --save_steps 10000 \
    --samples_per_regression_task 10 \
    --samples_per_classification_task 10 \
    --num_train_epochs 3

