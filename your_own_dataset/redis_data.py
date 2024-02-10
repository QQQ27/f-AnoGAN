import torch
import redis
import pickle
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys


def Gereping2Redis(img_folder, mask_folder, redis_db=0, redis_port=6379, int16=False):
    if not redis_port:
        redis_port = 6379
    if not redis_db:
        redis_db = 0
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


class GerepingRedisDataset(torch.utils.data.Dataset):
    def __init__(self, opt, mode="train", val_ratio=0.3, seed=7923, transform=None):
        super().__init__()
        """
        redis 键值说明"{"abnormal"/"normal"}&{img_name("1#SK1017_2_1)}"
        对应的值以 img序列化，mask序列化存放(normal 的mask 为0)
        """
        redis_db = opt.redis_db
        redis_port = opt.redis_port
        if not redis_db:
            redis_db = 0
        if not redis_port:
            redis_port = 6379
        self.r = redis.Redis(host="localhost", port=redis_port, db=redis_db)
        self.mode = mode
        keys = list(map(lambda x: x.decode(), self.r.keys()))
        test_keys = list(filter(lambda x: "abnormal" in x, keys))
        train_val_keys = list(filter(lambda x: "normal" in x, keys))

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
        file_name = self.keys[index].split("&")[1]
        img_bytes, mask_bytes = self.r.lindex("%s" % self.keys[index], 0), self.r.lindex("%s" % self.keys[index], 1)
        img, mask = pickle.loads(img_bytes), pickle.loads(mask_bytes)
        if self.mode in ["train", "val", "valid"]:
            img = Image.fromarray(img)
            return self.transform(img), torch.tensor(0)
        else:
            img, mask = Image.fromarray(img), Image.fromarray(mask)
            return self.transform(img), self.transform(mask), file_name


def load_gereping_train(opt):
    img_dir = [os.path.join(opt.train_root, "normal", "cut_imgs")]
    mask_dir = [os.path.join(opt.train_root, "normal", "cut_masks")]
    Gereping2Redis(img_dir, mask_dir, redis_db=opt.redis_db, redis_port=opt.redis_port)
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])
    train_ds = GerepingRedisDataset(opt, transform=transform, mode="train")
    # train_ds = GerepingH5Normal(opt, transform, mode="train")
    numworkers = 8 if sys.platform == "linux" else 0
    train_dl = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=numworkers,
                          pin_memory=True)
    return train_dl


def load_gereping_val(opt):
    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(1), transforms.ToTensor(),
    ])
    val_ds = GerepingRedisDataset(opt, transform=transform, mode="val")
    # val_ds = GerepingH5Normal(opt, transform, mode="val")
    numworkers = 8 if sys.platform == "linux" else 0
    val_dl = DataLoader(val_ds, batch_size=opt.batch_size, num_workers=numworkers, shuffle=False, drop_last=False)
    return val_dl


def load_gereping_test(opt):
    # img_dir = [os.path.join(opt["dataroot"], "abnormal", "cut_imgs")]
    # mask_dir = [os.path.join(opt["dataroot"], "abnormal", "cut_masks")]
    # Gereping2Redis(img_dir, mask_dir, redis_db=opt["redis_db"], redis_port=opt["redis_port"])
    transform = transforms.Compose([
        transforms.Resize(opt["img_size"]),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])
    test_ds = GerepingRedisDataset(opt, transform=transform, mode="test")
    test_dl = DataLoader(test_ds, batch_size=opt["Batch_Size"], shuffle=False, drop_last=False)
    return test_dl

