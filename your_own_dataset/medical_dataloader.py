import glob
import shutil
import sys

import numpy as np
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import cv2, lmdb
import random
import redis, pickle
from tqdm import tqdm


def clear_redis_db(port, db):
    print("Clear redis db%d" % db)
    redis_client = redis.StrictRedis(host="localhost", port=port, db=db)
    redis_client.select(db)
    redis_client.flushdb()
    redis_client.close()


# ----------------------------------------------------------------------------------------------------------------------
def MedicalData2Redis(img_folder, redis_db=0, redis_port=6379, singlechannel=False):
    clear_redis_db(redis_port, redis_db)
    r = redis.Redis(host="localhost", port=redis_port, db=redis_db)
    if isinstance(img_folder, str):
        img_folder = [img_folder]
    img_files = []
    for img_d in img_folder:
        img_files += glob.glob(os.path.join(img_d, "*.png"))
    for img_file in tqdm(img_files):
        img = cv2.imread(os.path.join(img_d, img_file), -1) if singlechannel \
            else cv2.imread(os.path.join(img_d, img_file), 0)
        img_bytes = pickle.dumps(img)
        r.rpush(img_file.replace(":\\", ""), img_bytes)


class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, opt, transform=None, mode="train", val_ratio=0.3):
        super(MedicalDataset, self).__init__()
        self.opt = opt
        self.mode = mode
        if isinstance(self.opt.dataroot, str):
            self.opt.dataroot = list(self.opt.dataroot)
        self.files = []
        for data_path in self.opt.dataroot:
            self.files += glob.glob(os.path.join(data_path, "*.png"))
        if self.mode in ["train", 'val', 'valid']:
            random.seed(9093)
            random.shuffle(self.files)

            val_num = int(len(self.files) * val_ratio)
            if self.mode == "train":
                self.data_files = self.files[val_num:]
            else:
                self.data_files = self.files[:val_num]
        elif self.mode == "test":
            self.data_files = self.files

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.opt.isize),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        # TODO 返回 [img, mask] 组合
        img = Image.open(self.data_files[index]).convert("L")
        img = self.transform(img)
        return img, torch.tensor(0)


class MedicalDatasetLmdb(Dataset):
    def __init__(self, opt, transform=None, mode="train", val_ratio=0.3):
        super(MedicalDatasetLmdb, self).__init__()
        self.imgs_dir = opt.dataroot
        if isinstance(opt.dataroot, str):
            self.image_file = glob.glob(os.path.join(opt.dataroot, "*.png"))
        elif isinstance(opt.dataroot, list):
            self.image_file = []
            for img_dir in opt.dataroot:
                self.image_file += glob.glob(os.path.join(img_dir, "*.png"))
        else:
            raise TypeError("Unsupported type of 'imgs_dir'.")
        self.env = lmdb.open(opt.lmdb_env_path, max_readers=32, readonly=True, lock=False, readahead=False,
                             meminit=False)
        self.transforms = transform
        if mode in ["train", "val", "valid"]:
            random.seed(9093)
            random.shuffle(self.image_file)
            len_train = int(len(self.image_file) * (1 - val_ratio))
            if mode == "train":
                self.image_file = self.image_file[:len_train]
            else:
                self.image_file = self.image_file[len_train:]

    def __len__(self):
        return len(self.image_file)

    def __getitem__(self, i):
        with self.env.begin(write=False) as txn:
            image_bin = txn.get((self.image_file[i]).encode())
            image_buf = np.frombuffer(image_bin, dtype=np.uint8)
            img = cv2.imdecode(image_buf, -1)
            img = Image.fromarray(img)
            img = self.transforms(img)
        return img, torch.tensor(0)


class MedicalDatasetRedis(Dataset):
    def __init__(self, opt, mode="train", val_ratio=0.3, seed=7923, transform=None):
        super().__init__()
        self.r = redis.Redis(host="localhost", port=opt.redis_port, db=opt.redis_db)
        keys = list(map(lambda x: x.decode(), self.r.keys()))
        self.mode = mode
        if self.mode in ["train", "val", "valid"]:
            random.seed(seed)
            random.shuffle(keys)
            len_train = int(len(keys) * (1 - val_ratio))
            if self.mode == "train":
                self.keys = keys[:len_train]
            else:
                self.keys = keys[len_train:]
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((opt.isize, opt.iszie)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        img_bytes = self.r.lindex(self.keys[index], 0)
        img = pickle.loads(img_bytes)
        if self.mode in ["train", "val", "valid"]:
            img = Image.fromarray(img)
            return self.transform(img), torch.tensor(0)
        else:
            mask_bytes = self.r.lindex(self.keys[index], 1)
            mask = pickle.loads(mask_bytes)
            mask = Image.fromarray(mask)
            return self.transform(img), os.path.basename(self.keys[index]), self.transform(mask)


class MedicalTestData(Dataset):
    def __init__(self, opt, transform=None):
        super().__init__()
        """
        针对含有一个patient对应一个文件夹的数据集
        """
        assert not opt.isTrain
        self.folder_list = glob.glob(os.path.join(opt.dataroot, "*.png"))
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((opt.isize, opt.isize)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, index):
        # TODO  添加掩膜路径，同时返回图像和掩膜
        data = Image.open(self.folder_list[index]).convert("L")
        self.folder_list[index].replace("\\", "/")
        img_file = "/".join(self.folder_list[index].split("\\")[-2:])
        return self.transform(data), img_file


def load_medical_train(opt):
    transform = transforms.Compose([
        transforms.Resize((opt.isize, opt.isize)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    # train_ds = MedicalDataset(opt, transform=transform, mode="train")
    # train_ds = MedicalDatasetLmdb(opt, transform=transform, mode="train")
    MedicalData2Redis(opt.dataroot, redis_db=opt.redis_db, redis_port=opt.redis_port)
    train_ds = MedicalDatasetRedis(opt, transform=transform, mode="train")
    numworkers = 8 if sys.platform == "linux" else 0
    train_dl = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True, num_workers=numworkers,
                          pin_memory=True)
    return train_dl


def load_medical_val(opt):
    transform = transforms.Compose([
        transforms.Resize((opt.isize, opt.isize)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    # val_ds = MedicalDataset(opt, transform=transform, mode="val")
    val_ds = MedicalDatasetRedis(opt, transform=transform, mode="val")
    numworkers = 8 if sys.platform == "linux" else 0
    val_dl = DataLoader(val_ds, batch_size=opt.batchsize, shuffle=True, drop_last=False, num_workers=numworkers,
                        pin_memory=True)
    return val_dl


def load_medical_test(opt):
    transform = transforms.Compose([
        transforms.Resize((opt.isize, opt.isize)),
        transforms.ToTensor()
    ])
    test_ds = MedicalTestData(opt, transform=transform)
    numworkers = 8 if sys.platform == "linux" else 0
    test_dl = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False, num_workers=numworkers,
                         pin_memory=True)
    return test_dl
