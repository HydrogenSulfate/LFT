# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 我的模型
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3.7 -m torch.distributed.launch --nproc_per_node 4 train.py \
--model_name LFTImp_gridsample \
--angRes 5 \
--dropout 0.1 \
--scale_factor 2 \
--feat_unfold \
--cell_decode \
--batch_size 1 \
--global_batch_size 8 \
--amp \
--data_name HCI_new \
--num_workers 4 \
--path_for_train './data/LFSR_processed_rgb/data_for_train/' \
--lr 3e-4 \
--epoch 80 \
--save_epoch 10 \
--n_steps 30
# --random_sample True
# --epi_loss 0.5
# --use_pre_pth True \
# --path_pre_pth './log/SR_5x5_2x/LFTImp_gridsample_baseline/HCI_new/checkpoints/LFTImp_gridsample_5x5_2x_epoch_72_model.pth'





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

