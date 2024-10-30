export CUDA_VISIBLE_DEVICES=0
python evaluate_qwen2_vl.py \
    --model_path /mnt/data/CVPR2025/task1_data/Qwen2-VL-2B-Instruct  \
    --hdf5_path /mnt/data/CVPR2025/task1_data/test_no_classification.hdf5\
    --image_dir /mnt/data/CVPR2025/task1_data/images/images \
    --template_path /mnt/data/CVPR2025/task1_workspace/astro_llava_nv/template_vision_only_zero_shot_qwen2vl.json \
    --batch_size 48 \
    --num_workers 16 \
    --output_dir ./evaluation_results_qwen2_vl_2B