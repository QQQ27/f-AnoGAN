import shutil
import sys

import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torch.utils.data import Dataset, DataLoader
# from lib.data.datasets import get_cifar_anomaly_dataset
# from lib.data.datasets import get_mnist_anomaly_dataset
# from lib.data.datasets import get_mnist_anomaly_dataset
# from lib.data.anomaly_data import AbnomalyDataset
import torch
from PIL import Image
import cv2
from collections import deque
import random
import h5py
from tqdm import tqdm
import redis, pickle


class DataPrefetcher():
    def __init__(self, loader):
        # loader = list(loader)
        self.stream = torch.cuda.Stream()
        self.loader = iter(loader)
        self.preload()

    def preload(self):
        try:
            # self.next_input, self.next_target = next(self.loader)
            # self.next_sampler = next(self.loader)
            self.img, self.mask = next(self.loader)
        except StopIteration:
            # self.next_input = None
            # self.next_target = None
            self.next_sampler = None
            return
        with torch.cuda.stream(self.stream):
            self.img = self.img.to(device="cuda:0", dtype=torch.float32, non_blocking=True)
            self.mask = self.mask.to(device="cuda:0", dtype=torch.float32, non_blocking=True)
            # self.next_sampler = {
            #     'image': self.next_sampler["image"].to(device="cuda", dtype=torch.float32, non_blocking=True),
            #     "mask": self.next_sampler["mask"].to(device="cuda", dtype=torch.float32, non_blocking=True)}
            # self.next_input = self.next_input.cuda(non_blocking=True).float()
            # self.next_target = self.next_target.cuda(non_blocking=True).long()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        # sampler = self.next_sampler
        img, mask = self.img, self.mask
        # input = self.next_input
        # target = self.next_target
        self.preload()
        # return input, target
        return img, mask
        # return sampler


# --------------------------------------------------------------------------------------------------------------------------------
def clear_redis_db(port, db):
    redis_client = redis.StrictRedis(host="localhost", port=port, db=db)
    redis_client.select(db)
    if redis_client.dbsize() != 0:
        ans = input("the redis db %d is not null, are you sure you want to clear the db?" % db)
        if ans.lower() == "y" or ans.lower() == "yes":
            print("Clear redis db%d" % db)
            redis_client.flushdb()
            redis_client.close()
        else:
            sys.exit()
    else:
        print("Clear redis db%d" % db)
        redis_client.flushdb()
        redis_client.close()


def Gereping2Redis(img_folder, mask_folder, redis_db=0, redis_port=6379, int16=False):
    clear_redis_db(redis_port, redis_db)
    r = redis.Redis(host="localhost", port=redis_port, db=redis_db)
    if isinstance(img_folder, str) and isinstance(mask_folder, str):
        img_folder, mask_folder = [img_folder], [mask_folder]
    for img_d, mask_d in zip(img_folder, mask_folder):
        mode = "abnormal" if "abnormal" in img_d else "normal"
        for img_file in tqdm(os.listdir(img_d)):
            mask_file = img_file.replace("tif", "png")
            img = cv2.imread(os.path.join(img_d, img_file), -1) if int16 \
                else cv2.imread(os.path.join(img_d, img_file), 0)
            mask = cv2.imread(os.path.join(mask_d, mask_file), -1) if mode == "abnormal" else 0
            comp_name = os.path.splitext(mask_file)[0]
            img_bytes, mask_bytes = pickle.dumps(img), pickle.dumps(mask)
            r.rpush("%s" % "&".join([mode, comp_name]), img_bytes, mask_bytes)


def GerepingfromRedis(img_save_folder, mask_save_folder, redis_db=0, redis_port=6379):
    r = redis.Redis(host="localhost", port=6379)
    keys = list(map(lambda x: x.decode(), r.keys()))
    os.makedirs(img_save_folder, exist_ok=True)
    os.makedirs(mask_save_folder, exist_ok=True)
    for k in keys:
        img_bytes, mask_bytes = r.lindex("%s" % k, 0), r.lindex("%s" % k, 1)
        img, mask = pickle.loads(img_bytes), pickle.loads(mask_bytes)
        name = k.split("&")[-1]
        cv2.imwrite(os.path.join(img_save_folder, "%s.tif" % name), img)
        if mask == 0:
            mask = np.zeros_like(img).astype(np.uint8)
        cv2.imwrite(os.path.join(mask_save_folder, "%s.png" % name), mask)


# TODO
class GerepingRedisDataset(torch.utils.data.Dataset):
    def __init__(self, opt, mode="train", val_ratio=0.3, seed=7923, transform=None):
        super().__init__()
        """
        redis 键值说明"{"abnormal"/"normal"}&{img_name("1#SK1017_2_1)}"
        对应的值以 img序列化，mask序列化存放(normal 的mask 为0)
        """
        redis_db, redis_port = opt.redis_db, opt.redis_port
        self.r = redis.Redis(host="localhost", port=redis_port, db=redis_db)
        keys = list(map(lambda x: x.decode(), self.r.keys()))
        test_keys = list(filter(lambda x: "abnormal" in x, keys))
        train_val_keys = list(filter(lambda x: "normal" in x, keys))
        self.mode = mode
        import random
        random.seed(seed)
        random.shuffle(train_val_keys)
        random.shuffle(test_keys)

        if mode == "train":
            self.keys = train_val_keys[:int((1 - val_ratio) * len(train_val_keys))]
        elif mode == "val" or mode == "valid":
            self.keys = train_val_keys[int((1 - val_ratio) * len(train_val_keys)):]
        elif mode == "test":
            self.keys = test_keys
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(1),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        img_bytes, mask_bytes = self.r.lindex("%s" % self.keys[index], 0), self.r.lindex("%s" % self.keys[index], 1)
        img, mask = pickle.loads(img_bytes), pickle.loads(mask_bytes)
        file_name = self.keys[index].split("&")[-1] + ".png"
        if self.mode == "test":
            img, mask = Image.fromarray(img), Image.fromarray(mask)
            return self.transform(img), file_name, self.transform(mask)
        else:
            img = Image.fromarray(img)
            return self.transform(img), torch.tensor(mask)


if __name__ == "__main__":
    img_dir = [r"/data4/lq/data/gereping/0440/normal/cut_imgs"]
    mask_dir = [r"/data4/lq/data/gereping/0440/normal/cut_masks"]
    Gereping2Redis(img_dir, mask_dir, redis_db=0)


# --------------------------------------------------------------------------------------------------------------------------------


def create_val(fp, val_ratio=0.4):
    """
    :param fp: 数据集路径
    """
    train_fp = os.path.join(fp, "train", "normal")
    val_fp = train_fp.replace("train", "val")
    os.makedirs(val_fp, exist_ok=True)
    train_fl = os.listdir(train_fp)
    random.shuffle(train_fl)
    val_fl = train_fl[:int(len(train_fl) * val_ratio)]
    for f in val_fl:
        shutil.move(os.path.join(train_fp, f), os.path.join(val_fp, f))


def create_h5(fp, save_path, mode="train"):
    if isinstance(fp, list):
        fp = list(map(lambda f: list(map(lambda x: os.path.join(f, x), os.listdir(f))), fp))
        pic_path = []
        for f in fp:
            pic_path += f
    elif isinstance(fp, str):
        pic_path = list(map(lambda x: os.path.join(fp, x), os.listdir(fp)))
    else:
        raise TypeError("Unsupported type of 'fp'.")

    f = h5py.File(os.path.join(save_path, "%s.h5" % mode), "w")
    data_group = f.create_group("data")
    if mode == "train" or mode == "val":
        for i, p in enumerate(tqdm(pic_path, desc="creating %s.h5" % mode)):
            data_group["%d" % i] = cv2.imread(p, 0)
    elif mode == "test":
        f = h5py.File(os.path.join(save_path, "test.h5"), "w")
        data_group = f.create_group("data")
        fp_group = f.create_group("fp")
        for i, p in enumerate(tqdm(pic_path, desc="creating test.h5")):
            pic = np.expand_dims(cv2.imread(p, 0), axis=0)
            data_group["%d" % i] = pic
            fp_group["%d" % i] = p
    else:
        raise ValueError("Unsupported param of 'mode'.")
    f.close()
    return


def read_h5(fp):
    f = h5py.File(fp, "r")
    data, label = f["data"][:], f["label"][:]
    try:
        data_path = f["fp"][:]
    except KeyError:
        data_path = None
    f.close()
    # a = data_path[0].decode().split("\\")[-1]
    return data, label, data_path


def cut_imgs2h5(fp, save_path, mode="normal"):
    """
    将图像块转换为h5文件，加速训练
    """
    f = h5py.File(os.path.join(save_path, "%s.h5" % mode), "w")
    data, label = f.create_group("data"), f.create_group("label")
    if isinstance(fp, str):
        if mode == "normal":
            fp = os.path.join(fp, mode, "imgs")
            for i, img_file in enumerate(os.listdir(fp)):
                img = cv2.imread(os.path.join(fp, img_file), 0)
                data["%d" % i], label["%d" % i] = img, 0

    elif isinstance(fp, list):
        fp_ = list(map(lambda x: os.path.join(x, mode), fp))
        if mode == "normal":
            tmp = list(map(lambda x: list(map(lambda y: os.path.join(x, y), os.listdir(x))), fp_))
            img_files = []
            for l in tmp:
                img_files.extend(l)
            for i, img_file in enumerate(img_files):
                img = cv2.imread(img_file, 0)
                data["%d" % i], label["%d" % i] = img, 0


def load_gereping_train(opt):
    img_dir = [os.path.join(opt.train_root, "normal", "cut_imgs")]
    mask_dir = [os.path.join(opt.train_root, "normal", "cut_masks")]
    Gereping2Redis(img_dir, mask_dir, redis_db=opt.redis_db, redis_port=opt.redis_port)

    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(1),
        # transforms.RandomChoice([
        #     transforms.ColorJitter(brightness=0.8),
        #     transforms.ColorJitter(hue=0.5),
        #     transforms.ColorJitter(contrast=0.5),
        #     transforms.ColorJitter(saturation=0.5),
        #     transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.3, hue=0.5)], p=0.5),
        transforms.ToTensor(),
    ])
    train_ds = GerepingRedisDataset(opt, transform=transform, mode="train")
    # train_ds = GerepingH5Normal(opt, transform, mode="train")
    numworkers = 8 if sys.platform == "linux" else 0
    train_dl = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=numworkers,
                          pin_memory=True)
    # train_ds = GerepingInSingleDirect(opt, transform, mode="train")
    # train_dl = DataLoaderX(train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    return train_dl


def load_gereping_val(opt):
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(1),
        # transforms.RandomChoice([
        #     transforms.ColorJitter(brightness=0.8),
        #     transforms.ColorJitter(hue=0.5),
        #     transforms.ColorJitter(contrast=0.5),
        #     transforms.ColorJitter(saturation=0.5),
        #     transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.3, hue=0.5)], p=0.5),
        transforms.ToTensor(),
    ])
    val_ds = GerepingRedisDataset(opt, transform=transform, mode="val")
    # val_ds = GerepingH5Normal(opt, transform, mode="val")
    numworkers = 8 if sys.platform == "linux" else 0
    # test_ds = GerepingInSingleDirect(opt, mode="valid")
    val_dl = DataLoader(val_ds, batch_size=opt.batch_size, num_workers=numworkers, shuffle=False, drop_last=False)
    return val_dl


def load_gereping_test(opt):
    img_dir = [os.path.join(opt.dataroot, "abnormal", "cut_imgs")]
    mask_dir = [os.path.join(opt.dataroot, "abnormal", "cut_masks")]
    Gereping2Redis(img_dir, mask_dir, redis_db=opt.redis_db, redis_port=opt.redis_port)

    transform = transforms.Compose([
        # transforms.CenterCrop(opt.isize // 2),
        transforms.Resize(opt.img_size),
        # transforms.RandomHorizontalFlip(),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])
    test_ds = GerepingRedisDataset(opt, transform=transform, mode="test")
    # test_ds = GerepingInSingleDirect(opt, mode="valid")
    test_dl = DataLoader(test_ds, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    return test_dl


