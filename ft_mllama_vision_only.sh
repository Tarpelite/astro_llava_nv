export CUDA_VISIBLE_DEVICES=0
accelerate launch ft_mllama_vision_only.py \
    --model_path /mnt/data/CVPR2025/task1_data/Llama-3.2-11B-Vision-Instruct \
    --hdf5_path /mnt/data/CVPR2025/task1_data/train_no_classification.hdf5 \
    --image_dir /mnt/data/CVPR2025/task1_data/images/images \
    --output_dir /mnt/data/CVPR2025/task1_data/ft_mllama_vision_only_v1_1029 \
    --train_batch_size 2 \
    --logging_steps 1 \
    --num_train_epochs 3