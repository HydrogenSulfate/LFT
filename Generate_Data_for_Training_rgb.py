import argparse
import os
import os.path as osp
from pathlib import Path

import cv2
import h5py

from utils.imresize import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angRes_in", type=int, default=3, help="angular resolution")
    parser.add_argument("--angRes_out", type=int, default=5, help="angular resolution")
    parser.add_argument("--patchsize", type=int, default=256, help="slide window crop size")
    parser.add_argument("--time_step", type=int, default=3, help="angular resolution")

    parser.add_argument('--data_for', type=str, default='training', help='')
    parser.add_argument('--src_data_path', type=str, default='D:/BaiduNetdiskDownload/INRIA_Raytrix_video/', help='')
    parser.add_argument('--save_data_path', type=str, default='./', help='')

    return parser.parse_args()


def extract_lf_from_dir(args, frame_dir: str) -> np.ndarray:
    """extract light field images and decode, combine them into an ndarray.

    Args:
        args (_type_): _description_
        frame_dir (str): _description_

    Returns:
        np.ndarray: [U,V,H,W,C]
    """
    lf = []
    for u in range(args.angRes_out):
        tmp_row = []
        for v in range(args.angRes_out):
            im_path = f"{frame_dir}/sa_{u+1}_{v+1}.png"
            assert os.path.exists(im_path), f"{im_path} not exist."
            im = cv2.imread(im_path)[..., ::-1]
            tmp_row.append(im) # [h,w,c]
        tmp_row = np.stack(tmp_row) # [v,h,w,c]
        lf.append(tmp_row)
    lf = np.stack(lf) # [u,v,h,w,c]
    return lf


def main(args):
    """从raw光场视频数据集中生成triplet数据作为训练数据
    期望的src文件夹结构:

    src_data_path/
    |-- dataset_name/
        |-- training/
            |-- frame_1/
            |-- frame_2/
            |-- ...
        |-- testing/
            |-- frame_1/
            |-- frame_1/
            |-- ...

    Args:
        args (_type_): _description_
    """
    angRes_in, angRes_out = args.angRes_in, args.angRes_out
    time_step = args.time_step
    patchsize = args.patchsize
    stride = patchsize // 2

    ''' dir '''
    save_dir = Path(args.save_data_path + 'data_for_' + args.data_for) # save_data_path/data_for_training
    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir.joinpath('SR_' + 't' + str(time_step) + '_' + 'a' + str(angRes_in) + '_' + 'a' + str(angRes_out)) # save_data_path/data_for_training/SR_{t}_{ain}_{aout}
    save_dir.mkdir(exist_ok=True)

    src_datasets = os.listdir(args.src_data_path)
    print(src_datasets)
    for index_dataset in range(len(src_datasets)):
        if src_datasets[index_dataset] not in ['boxer', 'chess']:
            continue
        name_dataset = src_datasets[index_dataset] # 'boxer'
        sub_save_dir = save_dir.joinpath(name_dataset) # save_data_path/data_for_training/SR_{t}_{ain}_{aout}/boxer
        sub_save_dir.mkdir(exist_ok=True)

        src_sub_dataset = args.src_data_path + name_dataset + '/' + args.data_for + '/' # src_data_path/boxer/training/
        frame_dirs = os.listdir(src_sub_dataset)

        # 按时间戳从小到大
        frame_dirs.sort(key=lambda filename: int(filename.split('_')[-1]))

        idx_save = 0
        for t, frame_dir in enumerate(frame_dirs): # ['frame_0.h5', 'frame_1.h5', ...]
            if t + 2 * time_step >= len(frame_dirs):
                break
            # idx_scene_save = 0
            print('Generating training data of Frame_[%s] in Dataset [%s]......' %(frame_dir, name_dataset), end='\t')

            LF_pre = extract_lf_from_dir(args, osp.join(src_sub_dataset, frame_dirs[t])) # [ah,aw,c]
            LF_mid = extract_lf_from_dir(args, osp.join(src_sub_dataset, frame_dirs[t+time_step])) # [ah,aw,c]
            LF_nxt = extract_lf_from_dir(args, osp.join(src_sub_dataset, frame_dirs[t+time_step*2])) # [ah,aw,c]
            (U, V, _, _, _) = LF_mid.shape
            assert LF_pre.shape == LF_mid.shape == LF_nxt.shape, \
                f"LF_pre.shape({LF_pre.shape}) == LF_mid.shape({LF_mid.shape}) == LF_nxt.shape({LF_nxt.shape})"
            print(f"shape of current frame triplet is {LF_pre.shape}")
            # exit(0)
            if LF_pre.dtype != np.double:
                LF_pre = LF_pre.astype('double') / 255 # scale to [0,1]
            if LF_mid.dtype != np.double:
                LF_mid = LF_mid.astype('double') / 255 # scale to [0,1]
            if LF_nxt.dtype != np.double:
                LF_nxt = LF_nxt.astype('double') / 255 # scale to [0,1]

            # Extract sparse views of pre and nxt.
            LF_pre = LF_pre[::2, ::2]
            LF_nxt = LF_nxt[::2, ::2]

            (U, V, H, W, _) = LF_mid.shape
            (U0, V0, H, W, _) = LF_pre.shape
            num_crops = ((H - patchsize + 1) // stride) * ((W - patchsize + 1) // stride)
            # print(num_crops)
            for h in range(0, H - patchsize + 1, stride):
                for w in range(0, W - patchsize + 1, stride):
                    idx_save = idx_save + 1
                    # idx_scene_save = idx_scene_save + 1
                    Lr_pre_SAI_rgb = np.zeros((U0 * patchsize, V0 * patchsize, 3), dtype='single')
                    Hr_mid_SAI_rgb = np.zeros((U  * patchsize, V  * patchsize, 3), dtype='single')
                    Lr_nxt_SAI_rgb = np.zeros((U0 * patchsize, V0 * patchsize, 3), dtype='single')
                    crop_rc = np.array([h, w]).astype('int64')
                    # process Hr middle frame
                    for u in range(U):
                        for v in range(V):
                            Hr_mid_crop_rgb = LF_mid[u, v, h: h + patchsize, w: w + patchsize,:]
                            Hr_mid_SAI_rgb[u * patchsize : (u+1) * patchsize, v * patchsize: (v+1) * patchsize] = Hr_mid_crop_rgb
                            pass
                        pass

                    # process lr previous and next frame
                    for u in range(U0):
                        for v in range(V0):
                            Lr_pre_crop_rgb = LF_pre[u, v, h: h + patchsize, w: w + patchsize,:]
                            Lr_nxt_crop_rgb = LF_nxt[u, v, h: h + patchsize, w: w + patchsize,:]
                            Lr_pre_SAI_rgb[u * patchsize : (u+1) * patchsize, v * patchsize: (v+1) * patchsize] = Lr_pre_crop_rgb
                            Lr_nxt_SAI_rgb[u * patchsize : (u+1) * patchsize, v * patchsize: (v+1) * patchsize] = Lr_nxt_crop_rgb
                            pass
                        pass

                    # save
                    file_name = [str(sub_save_dir) + '/' + '%06d'%idx_save + '.h5']
                    with h5py.File(file_name[0], 'w') as hf:
                        hf.create_dataset('Lr_pre_SAI_rgb', data=Lr_pre_SAI_rgb, dtype='single')
                        hf.create_dataset('Hr_mid_SAI_rgb', data=Hr_mid_SAI_rgb, dtype='single')
                        hf.create_dataset('Lr_nxt_SAI_rgb', data=Lr_nxt_SAI_rgb, dtype='single')
                        hf.create_dataset('crop_r0c0', data=crop_rc, dtype='int')
                        hf.close()
                        pass
                    exit(0)
                    pass
                pass
            print('(%d, %d, %d) training trpilet have been processed/n' % (t, t+time_step, t+2*time_step))
        pass
    pass



if __name__ == '__main__':
    args = parse_args()

    main(args)
