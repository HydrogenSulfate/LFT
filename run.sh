# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=1,2

python3.7 -m torch.distributed.launch --nproc_per_node 2 train.py \
--model_name LFTImp \
--angRes 5 \
--scale_factor 2 \
--batch_size 1 \
--global_batch_size 8 \
--amp \
--data_name HCI_new \
--num_workers 4

