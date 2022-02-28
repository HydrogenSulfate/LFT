import os
import random

import h5py
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from utils.utils import make_coord

# class TrainSetDataLoader(Dataset):
#     def __init__(self, args):
#         super(TrainSetDataLoader, self).__init__()
#         self.dataset_dir = args.path_for_train + 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + \
#                            str(args.scale_factor) + 'x/'

#         if args.data_name == 'ALL':
#             self.data_list = os.listdir(self.dataset_dir)
#         else:
#             self.data_list = [args.data_name]

#         self.file_list = []
#         for data_name in self.data_list:
#             tmp_list = os.listdir(self.dataset_dir + data_name)
#             for index, _ in enumerate(tmp_list):
#                 tmp_list[index] = data_name + '/' + tmp_list[index]

#             self.file_list.extend(tmp_list)

#         self.item_num = len(self.file_list)

#     def __getitem__(self, index):
#         file_name = [self.dataset_dir + self.file_list[index]]
#         with h5py.File(file_name[0], 'r') as hf:
#             data_SAI_y = np.array(hf.get('Lr_SAI_y'))   # Lr_SAI_y
#             label_SAI_y = np.array(hf.get('Hr_SAI_y'))  # Hr_SAI_y
#             data_SAI_y, label_SAI_y = augmentation(data_SAI_y, label_SAI_y)
#             data_SAI_y = ToTensor()(data_SAI_y.copy())
#             label_SAI_y = ToTensor()(label_SAI_y.copy())

#         return data_SAI_y, label_SAI_y

#     def __len__(self):
#         return self.item_num


class TrainSetDataLoader(Dataset):
    """RGB train dataset

    Args:
        Dataset (Dataset): LF dataset.
    """
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.dataset_dir = args.path_for_train + 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + \
            str(args.scale_factor) + 'x/'
        self.angRes = args.angRes
        self.random_sample = args.random_sample

        if args.data_name == 'ALL':
            self.data_list = os.listdir(self.dataset_dir)
        else:
            self.data_list = [args.data_name]
        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            data_SAI_rgb = np.array(hf.get('Lr_SAI_rgb')).transpose(2, 1, 0)  # [uh,vw,3]
            label_SAI_rgb = np.array(hf.get('Hr_SAI_rgb')).transpose(2, 1, 0)  # [uh',vw',3]
            data_SAI_rgb, label_SAI_rgb = augmentation(data_SAI_rgb, label_SAI_rgb)
            data_SAI_rgb = ToTensor()(data_SAI_rgb.copy())  # [3,uh,vw]
            label_SAI_rgb = ToTensor()(label_SAI_rgb.copy())  # [3,uh',vw']

        hr_coord = make_coord(
            [label_SAI_rgb.shape[-2] // self.angRes, label_SAI_rgb.shape[-1] // self.angRes],
            flatten=False
        )  # [h',w',2]

        if self.random_sample is True:  # 空间像素随机打乱
            _h, _w, _c = hr_coord.shape
            sample_lst = np.random.choice(
                _h * _w, _h * _w, replace=False)
            hr_coord = rearrange(hr_coord, 'h w c -> (h w) c', h=_h, w=_w, c=_c)
            hr_coord = hr_coord[sample_lst]
            hr_coord = rearrange(hr_coord, '(h w) c -> h w c', h=_h, w=_w, c=_c)

            label_SAI_rgb = rearrange(label_SAI_rgb, 'c (u h) (v w) -> c u v (h w)', u=self.angRes, v=self.angRes)
            label_SAI_rgb = label_SAI_rgb[:, :, :, sample_lst]
            label_SAI_rgb = rearrange(label_SAI_rgb, 'c u v (h w) -> c (u h) (v w)', h=_h, w=_w)

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / label_SAI_rgb.shape[-2]  # 一个cell的高
        cell[:, 1] *= 2 / label_SAI_rgb.shape[-1]  # 一个cell的宽


        return {
            'inp': data_SAI_rgb,
            'gt': label_SAI_rgb,
            'coord': hr_coord,
            'cell': cell
        }

    def __len__(self):
        return self.item_num


def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    dataset_dir = args.path_for_test + 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + \
                  str(args.scale_factor) + 'x/'
    data_list = os.listdir(dataset_dir)

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name)
        length_of_tests += len(test_Dataset)

        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name='ALL'):
        super(TestSetDataLoader, self).__init__()
        self.dataset_dir = args.path_for_test + 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + \
            str(args.scale_factor) + 'x/'
        self.data_list = [data_name]

        self.angRes = args.angRes
        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            data_SAI_rgb = np.array(hf.get('Lr_SAI_rgb')).transpose(2, 1, 0)  # [h,w,3]
            label_SAI_rgb = np.array(hf.get('Hr_SAI_rgb')).transpose(2, 1, 0)  # [h',w',3]
            data_SAI_rgb = ToTensor()(data_SAI_rgb.copy())  # [3,h,w]
            label_SAI_rgb = ToTensor()(label_SAI_rgb.copy())  # [3,h',w']

        hr_coord = make_coord([label_SAI_rgb.shape[-2] // self.angRes, label_SAI_rgb.shape[-1] // self.angRes], flatten=False)  # [h',w',2]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / label_SAI_rgb.shape[-2]  # 一个cell的高
        cell[:, 1] *= 2 / label_SAI_rgb.shape[-1]  # 一个cell的宽

        return {
            'inp': data_SAI_rgb,
            'gt': label_SAI_rgb,
        }

    def __len__(self):
        return self.item_num


def flip_SAI(data, angRes):
    if len(data.shape) == 2:
        H, W = data.shape
        data = data.reshape(H, W, 1)

    H, W, C = data.shape
    data = data.reshape(angRes, H // angRes, angRes, W // angRes, C)  # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)

    return data


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1, :]
        label = label[:, ::-1, :]
    if random.random() < 0.5:  # flip along H-U direction
        data = data[::-1, :, :]
        label = label[::-1, :, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0, 2)
        label = label.transpose(1, 0, 2)
    return data, label
