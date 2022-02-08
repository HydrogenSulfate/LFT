# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 我的模型
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3.7 -m torch.distributed.launch --nproc_per_node 4 train.py \
--model_name LFT \
--angRes 5 \
--scale_factor 2 \
--batch_size 1 \
--global_batch_size 8 \
--amp \
--data_name HCI_new \
--num_workers 4 \
--path_for_train './data/LFSR_processed_y/data_for_train/'
# --lr 3e-4 \
# --epoch 80 \



# LFT模型
# export CUDA_VISIBLE_DEVICES=1,2
# python3.7 -m torch.distributed.launch --nproc_per_node 2 train.py \
# --model_name LFT \
# --angRes 5 \
# --scale_factor 2 \
# --batch_size 1 \
# --global_batch_size 8 \
# --amp \
# --data_name HCI_new \
# --num_workers 4

