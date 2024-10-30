export CUDA_VISIBLE_DEVICES=0

accelerate launch  ft_qwen2_xval.py \
  --model_path /mnt/data/CVPR2025/task1_data/Qwen2-VL-2B-Instruct  \
  --train_hdf5_path /mnt/data/CVPR2025/task1_data/train_no_classification.hdf5 \
  --eval_hdf5_path  /mnt/data/CVPR2025/task1_data/test_no_classification.hdf5 \
  --image_dir /mnt/data/CVPR2025/task1_data/images/images \
  --template_path /mnt/data/CVPR2025/task1_workspace/astro_llava_nv/template_vision_only_zero_shot_qwen2vl.json \
  --output_dir /mnt/data/CVPR2025/task1_data/ckpts/ft_qwen2vl_2B_xval_v02_heavy_head\
  --eval_steps 100 \
  --eval_samples 80 \
  --save_steps 10000 \
  --train_batch_size 8 \
  --eval_batch_size 16 