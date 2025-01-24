import numpy as np 
import os
from os.path import join, split, isdir, isfile, abspath
import torch
from PIL import Image
import random
import collections
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2

transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
untransform = transforms.Compose([
    transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
])


class SkiTBDataset(Dataset):

    def __init__(self, root_dir, test, transform=None, untransform=None, output_size=(400, 400), set_augmentation=1):
        index_file = root_dir + ('test_idx.txt' if test else 'train_idx.txt')
        file_lines = [line.rstrip('\n') for line in open(index_file)]
        self.image_path = [join(root_dir, 'images', i + ".jpg") for i in file_lines]
        self.data_path = [join(root_dir, 'labels', i + ".txt") for i in file_lines]
        self.test = test
        self.output_size = output_size # (w, h)
        self.transform = transform
        self.untransform = untransform
        self.augmentation = set_augmentation if set_augmentation > 0 and not test else 1
    
    def __getitem__(self, item):

        if self.augmentation != 1:
            index = item // self.augmentation
            use_augmentation = item % self.augmentation != 0
        else:
            index = item
            use_augmentation = False

        path_img = self.image_path[index]
        path_label = self.data_path[index]

        assert isfile(path_img), path_img
        image = Image.open(path_img).convert('RGB')

        img_width, img_height = image.size
        output_width, output_height = self.output_size

        with open(path_label, 'r') as f:
            # line file order x1,x2,y1,y2
            line = [float(point) for point in f.readline().split(',')[0:4]]
            # line new order x1,y1,x2,y2
            line = [line[0], line[2], line[1], line[3]]
        
        # random crop the image
        if use_augmentation and random.random() < 0.5:
            crop_percent = random.uniform(0.1, 0.5)
            width_offset = int(img_width * crop_percent)
            height_offset = int(img_height * crop_percent)

            min_y, max_y = (line[1], line[3]) if line[1] < line[3] else (line[3], line[1])
            min_x, max_x = (line[0], line[2]) if line[0] < line[2] else (line[2], line[0])

            top = height_offset if height_offset < min_y else min_y
            left = width_offset if width_offset < min_x else min_x
            height = img_height - height_offset if img_height-height_offset > max_y else max_y
            width = img_width - width_offset if img_width-width_offset > max_x else max_x

            image = transforms.functional.crop(image, top, left, height, width)
                #size=(output_height, output_width),
                #interpolation=transforms.InterpolationMode.NEAREST
            # )
            # adjust line coordinate after crop
            line = [
                line[0] - left,
                line[1] - top,
                line[2] - left,
                line[3] - top,
            ]
            img_width, img_height = image.size
        
        # random color jittering
        if use_augmentation and random.random() < 0.5:
            image = transforms.ColorJitter(
                brightness=(0.2, 1.7),
                contrast=(0.2, 1.7),
                saturation=(0.2, 1.7)
            )(image)

        # to tensor
        if self.transform is not None:
            image = self.transform(image)
            line = [
                ((line[0]*output_width) / img_width) / output_width,
                ((line[1]*output_height) / img_height) / output_height,
                ((line[2]*output_width) / img_width) / output_width,
                ((line[3]*output_height) / img_height) / output_height
            ]
            
        # random horizontal filp the image
        if use_augmentation and random.random() < 0.5:
            image = transforms.functional.hflip(image)
            line[0] = 1 - line[0]
            line[2] = 1 - line[2]
        
        # random blur the image
        if use_augmentation and random.random() < 0.5:
            random_kernel_size = random.choice([7, 11, 17])
            image = transforms.functional.gaussian_blur(image, kernel_size=random_kernel_size)
        
        line = torch.tensor(line)
        
        return image, line, path_img.split('/')[-1]

    def __len__(self):
        return len(self.image_path) * self.augmentation

    # def collate_fn(self, batch):
    #     images, lines, names, flipped = list(zip(*batch))
    #     images = torch.stack([image for image in images])
    #     lines = torch.stack([line for line in lines])

    #     return images, lines, names, flipped

def get_loader(root_dir, test, batch_size, shuffle, num_workers, set_augmentation=1):
    dataset = SkiTBDataset(root_dir, test, transform=transform, untransform=untransform, set_augmentation=set_augmentation)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=not test,
        num_workers=num_workers,
    )

        
