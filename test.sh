# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=3

python3.7 test.py \
--model_name LFT \
--angRes 5 \
--scale_factor 2 \
--use_pre_pth True \
--path_for_test './data/LFSR_processed_y/data_for_test/' \
--path_pre_pth './log/SR_5x5_2x/LFT/HCI_new/checkpoints/LFT_5x5_2x_epoch_50_model.pth'
