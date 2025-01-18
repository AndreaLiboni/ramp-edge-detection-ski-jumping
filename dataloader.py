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


class SkiTBDataset(Dataset):

    def __init__(self, root_dir, split='train', transform=None):
        index_file = root_dir + ('train_idx.txt' if split == 'train' else 'test_idx.txt')
        lines = [line.rstrip('\n') for line in open(index_file)]
        self.image_path = [join(root_dir, 'images', i + ".jpg") for i in lines]
        self.data_path = [join(root_dir, 'labels', i + ".txt") for i in lines]
        self.split = split
        self.transform = transform
        self.output_size = (400, 400) # (w, h)
    
    def __getitem__(self, item):

        path_img = self.image_path[item]
        path_label = self.data_path[item]

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
        flipped = False
        if self.split == 'train' and random.random() < 0.5:
            flipped = True
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
        
        if self.split == 'train' and random.random() < 0.5:
            #flipped = True
            image = transforms.functional.hflip(image)
            line[0] = 1 - line[0]
            line[2] = 1 - line[2]
        
        # random blur the image
        if self.split == 'train' and random.random() < 0.5:
            random_kernel_size = random.choice([7, 11, 17])
            image = transforms.functional.gaussian_blur(image, kernel_size=random_kernel_size)
        
        # random crop the image
        
        line = torch.tensor(line)
        
        return image, line, path_img.split('/')[-1], flipped

    def __len__(self):
        return len(self.image_path)

    def collate_fn(self, batch):
        images, lines, names, flipped = list(zip(*batch))
        images = torch.stack([image for image in images])
        lines = torch.stack([line for line in lines])

        return images, lines, names, flipped


class SemanLineDataset(Dataset):

    def __init__(self, root_dir, label_file, split='train', transform=None, t_transform=None):
        lines = [line.rstrip('\n') for line in open(label_file)]
        self.image_path = [join(root_dir, i+".jpg") for i in lines]
        self.data_path = [join(root_dir, i+".npy") for i in lines]
        self.split = split
        self.transform = transform
        self.t_transform = t_transform
    
    def __getitem__(self, item):

        assert isfile(self.image_path[item]), self.image_path[item]
        image = Image.open(self.image_path[item]).convert('RGB')

        data = np.load(self.data_path[item], allow_pickle=True).item()
        hough_space_label8 = data["hough_space_label8"].astype(np.float32)
        if self.transform is not None:
            image = self.transform(image)
            
        hough_space_label8 = torch.from_numpy(hough_space_label8).unsqueeze(0)
        gt_coords = data["coords"]
        
        if self.split == 'val':
            return image, hough_space_label8, gt_coords, self.image_path[item].split('/')[-1]
        elif self.split == 'train':
            return image, hough_space_label8, gt_coords, self.image_path[item].split('/')[-1]

    def __len__(self):
        return len(self.image_path)

    def collate_fn(self, batch):
        images, hough_space_label8, gt_coords, names = list(zip(*batch))
        images = torch.stack([image for image in images])
        hough_space_label8 = torch.stack([hough_space_label for hough_space_label in hough_space_label8])

        return images, hough_space_label8, gt_coords, names

class SemanLineDatasetTest(Dataset):

    def __init__(self, root_dir, label_file, transform=None, t_transform=None):
        lines = [line.rstrip('\n') for line in open(label_file)]
        self.image_path = [join(root_dir, i+".jpg") for i in lines]
        self.transform = transform
        self.t_transform = t_transform
        
    def __getitem__(self, item):

        assert isfile(self.image_path[item]), self.image_path[item]
        image = Image.open(self.image_path[item]).convert('RGB')
        w, h = image.size
        if self.transform is not None:
            image = self.transform(image)
            
        return image, self.image_path[item].split('/')[-1], (h, w)

    def __len__(self):
        return len(self.image_path)

    def collate_fn(self, batch):
        images, names, sizes = list(zip(*batch))
        images = torch.stack([image for image in images])
    
        return images, names, sizes

def get_loader(root_dir, label_file, batch_size, img_size=0, num_thread=4, pin=True, test=False, split='train'):
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SkiTBDataset(root_dir, transform=transform, split=split)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=num_thread,
                                    pin_memory=pin, collate_fn=dataset.collate_fn)

        
