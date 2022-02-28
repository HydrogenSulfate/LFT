# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=3

python3.7 test.py \
--model_name LFTImp_gridsample \
--angRes 5 \
--scale_factor 2 \
--feat_unfold \
--cell_decode \
--use_pre_pth True \
--path_for_test './data/LFSR_processed_rgb/data_for_test/' \
--path_pre_pth './log/SR_5x5_2x/LFTImp_gridsample/ALL/checkpoints/LFTImp_gridsample_5x5_2x_best_model_37.32.pth'
