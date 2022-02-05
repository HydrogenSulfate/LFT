import os
from torch.utils.data import Dataset
from skimage import metrics
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader
from utils import *


class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.dataset_dir = args.path_for_train + 'SR_' + str(args.angRes) + 'x' + str(args.angRes) + '_' + \
                           str(args.scale_factor) + 'x/'

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
        self.angRes = args.angRes

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            data_SAI_rgb = np.array(hf.get('Lr_SAI_rgb'))   # Lr_SAI_rgb
            data_SAI_rgb = data_SAI_rgb.transpose(1, 2, 0) # [ah,aw,c]
            label_SAI_rgb = np.array(hf.get('Hr_SAI_rgb'))  # Hr_SAI_rgb
            label_SAI_rgb = label_SAI_rgb.transpose(1, 2, 0) # [ah,aw,c]
            H_hr, W_hr = label_SAI_rgb.shape[:2]
            H_hr = H_hr // self.angRes
            W_hr = W_hr // self.angRes

            data_SAI_rgb, label_SAI_rgb = augmentation_rgb(data_SAI_rgb, label_SAI_rgb)
            coord_hr = utils.make_coord([H_hr, W_hr]) # [H'W',2]
            data_SAI_rgb = ToTensor()(data_SAI_rgb.copy())
            label_SAI_rgb = ToTensor()(label_SAI_rgb.copy())

        return data_SAI_rgb, label_SAI_rgb, coord_hr

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

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)
        self.angRes = args.angRes

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_rgb = np.array(hf.get('Lr_SAI_rgb'))
            Hr_SAI_rgb = np.array(hf.get('Hr_SAI_rgb'))
            Lr_SAI_rgb = np.transpose(Lr_SAI_rgb, (2, 1, 0))
            Hr_SAI_rgb = np.transpose(Hr_SAI_rgb, (2, 1, 0))

            H_hr, W_hr = label_SAI_rgb.shape[:2]
            H_hr = H_hr // self.angRes
            W_hr = W_hr // self.angRes
            coord_hr = utils.make_coord([H_hr, W_hr]) # [H'W',2]

        Lr_SAI_rgb = ToTensor()(Lr_SAI_rgb.copy())
        Hr_SAI_rgb = ToTensor()(Hr_SAI_rgb.copy())

        return Lr_SAI_rgb, Hr_SAI_rgb, coord_hr

    def __len__(self):
        return self.item_num


def flip_SAI(data, angRes):
    if len(data.shape) == 2:
        H, W = data.shape
        data = data.reshape(H, W, 1)

    H, W, C = data.shape
    data = data.reshape(angRes, H//angRes, angRes, W//angRes, C)  # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)

    return data


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    return data, label

def augmentation_rgb(data, label):
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
