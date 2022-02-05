# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=3

python test.py \
--model_name LFTImp \
--angRes 5 \
--scale_factor 2 \
--use_pre_pth True \
--path_pre_pth '.log/SR_5x5_2x/LFTImp/ALL/checkpoints/LFTImp_5x5_2x_epoch_01_model.pth'