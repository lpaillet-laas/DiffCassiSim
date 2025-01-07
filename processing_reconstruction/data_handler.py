import os
import torch
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
import random
from lightning import LightningDataModule


class CubesDataset(Dataset):
    def __init__(self, data_dir, crop_size = 512, augment=True):
        self.data_dir = data_dir
        self.augment_ = augment
        self.crop_size = crop_size
        self.data_file_names = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.data_file_names)

    def __getitem__(self, idx):

        cube = self.load_hyperspectral_data(idx) # H x W x lambda

        if self.augment_:
            cube = self.augment(cube, self.crop_size) # lambda x H x W
        else:
            cube = torch.from_numpy(np.transpose(cube, (2, 0, 1))).float()[:,:self.crop_size,:self.crop_size] # lambda x H x W
        
        return cube

    def load_hyperspectral_data(self, idx):
        file_path = os.path.join(self.data_dir, self.data_file_names[idx])
        data = sio.loadmat(file_path)
        if "img_expand" in data:
            cube = data['img_expand'] / 65536.
        elif "img" in data:
            cube = data['img'] / 65536.
        cube = cube.astype(np.float32) # H x W x lambda

        return cube
    
    def augment(self, img, crop_size = 512):
        h, w, _ = img.shape

        # Randomly crop
        x_index = np.random.randint(0, h - crop_size)
        y_index = np.random.randint(0, w - crop_size)
        processed_data = np.zeros((crop_size, crop_size, 28), dtype=np.float32)
        processed_data = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = torch.from_numpy(np.transpose(processed_data, (2, 0, 1))).float()

        # Randomly flip and rotate
        processed_data = arguement_1(processed_data)

        return processed_data
    
class CubesDatasetTest(Dataset):
    def __init__(self, data_dir, augment=True, crop_size = 512):
        self.data_dir = data_dir
        self.augment_ = augment
        self.crop_size = crop_size
        self.data_file_names = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.data_file_names)

    def __getitem__(self, idx):

        cube = self.LoadTest(idx)
        
        return cube[:,:self.crop_size,:self.crop_size]
    
    #TODO Use this with KAIST dataset as a test function
    def LoadTest(self, idx):
        file_path = os.path.join(self.data_dir, self.data_file_names[idx])

        img = sio.loadmat(file_path)['img']
        test_data = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        return test_data

    
class CubesDataModule(LightningDataModule):
    def __init__(self, data_dir_train, data_dir_test, batch_size, crop_size=512, num_workers=1, augment=True):
        super().__init__()
        self.data_dir_train = data_dir_train
        self.data_dir_test = data_dir_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = CubesDataset(self.data_dir_train, augment=augment, crop_size=crop_size)
        self.test_dataset = CubesDatasetTest(self.data_dir_test, augment=augment, crop_size=crop_size)

    def setup(self, stage=None):
        dataset_size = len(self.dataset)
        train_size = int(0.79 * dataset_size)
        val_size = int(0.2 * dataset_size)
        test_size = dataset_size - train_size - val_size

        self.train_ds, self.val_ds, self.test_ds = random_split(self.dataset, [train_size, val_size, test_size])

        self.test_ds = self.test_dataset
        self.predict_ds = self.test_ds
    
    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_ds,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False)

def arguement_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))

    return x
