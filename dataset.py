import numpy as np
import torch
import os
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms, utils
import cv2, glob


class NMNIST(Dataset):
    def __init__(self, train, step, dt, path=None):
        """N-MNIST数据集初始化。如果没有预处理后的.npy文件，需要手动调用一次preprocessing函数进行预处理。

        Args:
            train (bool): 是否为训练集。为False则为测试集。
                If "True", it means that's the traning set. Otherwise it's the testing set.
            step (int): 超参数step数。代表SNN模拟的时间步数。
                How many steps that SNNs will simulate.
            dt (int): 超参数dt，单位ms，代表一个时间步所代表的实际时间。N-MNIST数据集带有时间戳，所以需要此参数。
                How long does a step really required. It's needed because N-MNIST dataset includes time stamp.
            path (string, optional): 参数为预处理后的.npy数据文件路径时，可以预加载.npy文件。默认为None。
                Set this to the path of data file (with ".npy" suffix) to accelerate data loading. Default to None.
        """
        super(NMNIST, self).__init__()
        self.step = step
        self.path = path
        self.train = train
        self.dt = dt
        self.win = step * dt
        self.len = 60000
        if train == False:
            self.len = 10000
        self.eventflow = np.zeros(shape=(self.len, 2, 34, 34, self.step))
        self.label = np.zeros(shape=(self.len, 10))
        
        if path is not None:
            self.eventflow = np.load(path + "/data.npy")
            self.label = np.load(path+"/label.npy")
        # abandon first zero item
        #self.label = self.label[1:]
        #self.eventflow = self.eventflow[1:, :, :, :, :]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        """Dataset类的重载方法。返回指定idx位置的数据。可以在此处进行必要的预处理，但可能会拖慢性能。

        Args:
            idx (int/Tensor/list): 数据index。

        Returns:
            (x, y): 数据/标签组成的元组。
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.eventflow[idx, ...].astype(np.float32)     # 某些情况下可能对数据格式有要求（MSELoss）
        y = self.label[idx].astype(np.float32)                
        return (x, y)

    def preprocessing(self, src_path, save_path=None):
        """此函数用于预处理原生的N-MNIST数据。指定存储路径的话，函数会在指定目录下存储相应的.npy文件，方便下次直接调用。需要注意的是，此文件仅用于加速同一份数据的读取。如果改变了如下超参数：dt（每个step的时间宽度）、step（输入的step总数）以及极性等其他改变输入数据的参数，则需要自己额外进行预处理或是生成新的.npy文件。

        Args:
            src_path (string): 源文件位置。
            save_path (string, optional): 存储目标路径。不指定时，不会存储文件。默认为None。
        """
        filenum = 0
        for num in range(10):
            dir = os.path.join(src_path, str(num))
            files = os.listdir(dir)
            for file in files:
                file_dir = os.path.join(dir, file)
                f = open(file_dir, 'rb')
                raw_data = np.fromfile(f, dtype=np.uint8)
                f.close()
                raw_data = np.uint32(raw_data)

                all_y = raw_data[1::5]
                all_x = raw_data[0::5]
                all_p = (raw_data[2::5] & 128) >> 7 #bit 7
                all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
                all_ts = np.uint32(np.around(all_ts / 1000))

                # data shape: (batch_size, channel/polar, height, width, time_window)
                win_indices = np.where(all_ts < self.win) # select the data in simulate window
                win_indices = win_indices[0] # squeeze tuple
                for i in range(len(win_indices)): 
                    index = int(win_indices[i])
                    polar = 0 # Drop the polar
                    self.eventflow[filenum, polar, all_x[index], all_y[index], int(all_ts[index] / self.dt)] = 1
                    # 1 for an event, 0 for nothing
                self.label[filenum] = np.eye(10)[num] # one-hot label

                filenum += 1
                
            print("Done file:" + str(num))
        
        if save_path is not None:
            field = "Train" if self.train else "Test"
            np.save("./data/NMNIST_npy/"+field+"/data.npy", self.eventflow)
            np.save("./data/NMNIST_npy/"+field+"/label.npy", self.label)




#class DatasetWrapper:
