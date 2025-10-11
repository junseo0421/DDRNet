CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29505 train_kd.py \
 --result_dir "/content/drive/MyDrive/DDRNet23s_kd_001" \
 --epochs 500 \
 --lr 5.e-4 \
 --scale_range [0.75,1.25] \
 --crop_size [1080,1920] \
 --batch_size 4 \
 --dataset_dir "/content/drive/MyDrive/SemanticDataset_lednet" \
