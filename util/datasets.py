# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import pandas as pd
from torchvision import datasets, transforms
import clip
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torch import nn
from PIL import Image
import glob
import torch
from sklearn.model_selection import train_test_split
import numpy as np


def get_classes(dir):
    classes = sorted(entry.name for entry in os.scandir(dir) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return class_to_idx

class CLIPDataset(Dataset):
    def __init__(self, root, transforms, is_train=True):

        self.root=root
        self.transforms=transforms
        label_mapping = {k: v for v, k in enumerate(os.listdir(self.root))}

        all_images = glob.glob(os.path.join(self.root,"*", "*"))
        all_images = [im for im in all_images if im.split(".")[-1]!="svg"]
        target = [label_mapping[path.split("/")[-2]] for path in all_images]
        np_images = np.array(all_images)
        np_target = np.array(target)

        x_train, x_test, y_train, y_test = train_test_split(np_images, np_target, test_size=0.3, random_state=32)
        x_train_real, x_val, y_train_real, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=32)
        if is_train:
            self.X = x_train_real
            self.Y = y_train_real
        else:
            self.X = x_val
            self.Y = y_val

        self.class_to_idx = get_classes(self.root)



    def __len__(self):
         return len(self.X)
    
    def __getitem__(self, idx):

        x = Image.open(self.X[idx]).convert("RGB")
        label  = self.Y[idx]

        if self.transforms:
            x = self.transforms(x)

        return x, torch.tensor(label)



class FishDataset4(Dataset):
    def __init__(self, root, transforms, is_train=True):

        self.root=root
        self.transforms=transforms
        label_mapping = {k: v for v, k in enumerate(os.listdir(self.root))}

        all_images = glob.glob(os.path.join(self.root,"*", "*"))
        all_images = [im for im in all_images if im.split(".")[-1]!="svg"]
        target = [label_mapping[path.split("/")[-2]] for path in all_images]
        np_images = np.array(all_images)
        np_target = np.array(target)

        x_train, x_test, y_train, y_test = train_test_split(np_images, np_target, test_size=0.5, random_state=32)
        x_train_real, x_val, y_train_real, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=32)
        if is_train:
            self.X = x_train_real
            self.Y = y_train_real
        else:
            self.X = x_val
            self.Y = y_val

    def __len__(self):
         return len(self.X)
    
    def __getitem__(self, idx):

        x = Image.open(self.X[idx]).convert("RGB")
        label  = self.Y[idx]

        if self.transforms:
            x = self.transforms(x)

        return x, torch.tensor(label)



class AgriClass(Dataset):
    def __init__(self, root, transforms, args, is_train=True):

        self.root=root
        self.transforms=transforms
        label_mapping = {k: v for v, k in enumerate(os.listdir(self.root))}

        all_images = glob.glob(os.path.join(self.root,"*", "*"))
        all_images = [im for im in all_images if im.split(".")[-1]!="svg"]
        target = [label_mapping[path.split("/")[-2]] for path in all_images]
        np_images = np.array(all_images)
        np_target = np.array(target)
        if args.eval and not is_train:
            x_test = np_images
            y_test = np_target
        else:
            x_train, x_val, y_train, y_val = train_test_split(np_images, np_target, test_size=0.1, random_state=32)

        if is_train:
            self.X = x_train
            self.Y = y_train
        else:
            if args.eval and not is_train:
                self.X = x_test
                self.Y = y_test
            else:
                self.X = x_val
                self.Y = y_val

    def __len__(self):
         return len(self.X)
    
    def __getitem__(self, idx):

        x = Image.open(self.X[idx]).convert("RGB")
        label  = self.Y[idx]

        if self.transforms:
            x = self.transforms(x)

        return x, torch.tensor(label)


class FishDataset10(Dataset):
    def __init__(self, root, transforms, is_train=True):

        self.root=root
        self.transforms=transforms
        if is_train:
            self.df  = pd.read_csv(os.path.join(self.root, "train.csv"))
        else:
            self.df  = pd.read_csv(os.path.join(self.root, "val.csv"))
        self.X = []
        self.Y = []
        for index, row in self.df.iterrows():
            self.X.append(os.path.join(self.root, row['ID'] + ".jpg"))
            self.Y.append(row['labels'])

    def __len__(self):
         return len(self.X)
    
    def __getitem__(self, idx):

        x = Image.open(self.X[idx])
        label  = self.Y[idx]

        if self.transforms:
            x = self.transforms(x)

        return x, torch.tensor(label)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset_type == "fish10":
        #dataset = datasets.ImageFolder(args.data_path, transform=transform)
        dataset = FishDataset10(args.data_path, transform, is_train=is_train)
    elif args.dataset_type == "fish4":
        dataset = FishDataset4(args.data_path, transform, is_train=is_train)
    elif args.dataset_type == "agri21":
        #root = os.path.join(args.data_path, 'train' if is_train else 'validation')
        if args.eval:
            root = os.path.join(args.data_path, 'validation')
            dataset = AgriClass(root, transform, args, is_train=is_train)
            return dataset

        root = os.path.join(args.data_path, 'train')
        dataset = AgriClass(root, transform, args, is_train=is_train)
        #dataset = datasets.ImageFolder(root, transform=transform)
    elif args.dataset_type == "agri24":
        if args.eval:
            root = os.path.join(args.data_path, 'test')
            dataset = AgriClass(root, transform, args, is_train=is_train)
            return dataset
        #root = os.path.join(args.data_path, 'train' if is_train else 'test')
        #dataset = datasets.ImageFolder(root, transform=transform)
        root = os.path.join(args.data_path, 'train')
        dataset = AgriClass(root, transform,args, is_train=is_train)

    elif args.dataset_type == "clip":
        dataset = CLIPDataset(args.data_path, transform, is_train=is_train)
    else:
        
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
