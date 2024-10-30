export CUDA_VISIBLE_DEVICES=0
python evaluate.py \
    --model_path /mnt/data/CVPR2025/task1_data/Llama-3.2-11B-Vision-Instruct  \
    --hdf5_path /mnt/data/CVPR2025/task1_data/test_no_classification.hdf5\
    --image_dir /mnt/data/CVPR2025/task1_data/images/images \
    --template_path template_vision_only_zero_shot.json \
    --batch_size 8 \
    --output_dir ./evaluation_results