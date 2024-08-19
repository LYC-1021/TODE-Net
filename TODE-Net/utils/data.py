import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

import os
from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random
import shutil


class IRSTD_Dataset(Data.Dataset):
    def __init__(self, args, mode='train'):
        
       dataset_dir = args.dataset_dir

       if mode == 'train':
          txtfile = 'img_idx/train.txt'
          #txtfile = 'img_idx/train_NUDT-SIRST.txt'
       elif mode == 'val':
          #txtfile = 'img_idx/test_NUDT-SIRST.txt'
          txtfile = 'img_idx/test.txt'

       self.list_dir = osp.join(dataset_dir, txtfile)
       self.imgs_dir = osp.join(dataset_dir, 'images')
       self.label_dir = osp.join(dataset_dir, 'masks')

       self.names = []
       with open(self.list_dir, 'r') as f:
           self.names += [line.strip() for line in f.readlines()]
       self.mode = mode
       self.crop_size = args.crop_size
       self.base_size = args.base_size
       self.transform = transforms.Compose([
           transforms.ToTensor(),
           #transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
           transforms.Normalize([-0.1246], [1.0923]), # mean and std
       ])
    
    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')     
        
        label_path = osp.join(self.label_dir, name+'.png')
        #img = Image.open(img_path).convert('RGB')
        img = Image.open(img_path).convert('L')
        mask = Image.open(label_path).convert('1')
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        
        img, mask = self.transform(img), transforms.ToTensor()(mask)
        #return img, mask,name
        return img, mask       #原本为这一行
    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):   #训练同步变换方法
        # random mirror
        if random.random() < 0.5:   # 有50%的概率执行以下操作
            img = img.transpose(Image.FLIP_LEFT_RIGHT)  # 将图像进行左右翻转
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT) # 将掩码也进行左右翻转
        crop_size = self.crop_size   # 获取预设的裁剪尺寸  
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))  # 随机生成一个长边尺寸，范围在base_size的一半到两倍之间  
        w, h = img.size    # 获取图像的宽和高 
        if h > w:           # 根据图像的长宽比调整图像的尺寸，确保长边为long_size，同时计算短边的尺寸
            oh = long_size    # 设置新的高度为long_size 
            ow = int(1.0 * w * long_size / h + 0.5)   # 根据原图的宽高比计算新的宽度  
            short_size = ow  # 短边尺寸为新计算的高度
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)   # 使用双线性插值方法调整图像尺寸
        mask = mask.resize((ow, oh), Image.NEAREST)  # 使用最近邻插值方法调整掩码尺寸  
        # pad crop
        if short_size < crop_size:   # 如果调整后的短边尺寸小于预设的裁剪尺寸
            padh = crop_size - oh if oh < crop_size else 0  # 计算高度需要填充的像素数  
            padw = crop_size - ow if ow < crop_size else 0   # 计算宽度需要填充的像素数
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0) # 对图像和掩码进行填充，填充颜色为黑色（0值）
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size  # 再次获取调整并可能填充后的图像尺寸 
        x1 = random.randint(0, w - crop_size)   # 随机选择一个裁剪的起始点x坐标
        y1 = random.randint(0, h - crop_size)    # 随机选择一个裁剪的起始点y坐标  
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))   # 根据选择的起始点和预设的裁剪尺寸对图像和掩码进行裁剪 
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:   # 有50%的概率执行以下操作
            img = img.filter(ImageFilter.GaussianBlur(   # 对图像进行高斯模糊处理，模糊半径为随机数 
                radius=random.random()))
        return img, mask


    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask
