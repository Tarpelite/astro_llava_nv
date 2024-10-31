export CUDA_VISIBLE_DEVICES=0

accelerate launch  ft_qwen2_full.py \
  --model_path /mnt/data/CVPR2025/task1_data/Qwen2-VL-2B-Instruct  \
  --train_hdf5_path /mnt/data/CVPR2025/task1_data/train_no_classification_addfeat.hdf5 \
  --eval_hdf5_path  /mnt/data/CVPR2025/task1_data/test_no_classification_addfeat.hdf5 \
  --image_dir /mnt/data/CVPR2025/task1_data/images/images \
  --template_path /mnt/data/CVPR2025/task1_workspace/astro_llava_nv/template_full_qwen2vl.json \
  --output_dir /mnt/data/CVPR2025/task1_data/ckpts/ft_qwen2vl_2B_full\
  --eval_steps 10 \
  --eval_samples 10 \
  --train_batch_size 8 \
  --eval_batch_size 1 