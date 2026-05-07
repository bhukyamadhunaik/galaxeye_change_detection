import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import random

class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=256):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        self.pre_dir = os.path.join(root_dir, 'pre-event')
        self.post_dir = os.path.join(root_dir, 'post-event')
        self.target_dir = os.path.join(root_dir, 'target')
        
        # We assume files have the same names across pre, post, target
        # Using post_dir to get list of files
        self.files = [os.path.basename(f) for f in glob.glob(os.path.join(self.post_dir, '*.tif'))]
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        filename = self.files[idx]
        
        pre_path = os.path.join(self.pre_dir, filename)
        post_path = os.path.join(self.post_dir, filename)
        target_path = os.path.join(self.target_dir, filename)
        
        pre_img = Image.open(pre_path).convert('RGB')
        post_img = Image.open(post_path).convert('L') # SAR is 1-channel
        target_img = Image.open(target_path)
        
        # Convert to numpy to process labels
        target_arr = np.array(target_img, dtype=np.uint8)
        
        # Label remapping
        # 0: Background -> 0 (No-Change)
        # 1: Intact -> 0 (No-Change)
        # 2: Damaged -> 1 (Change)
        # 3: Destroyed -> 1 (Change)
        remapped_target = np.zeros_like(target_arr)
        remapped_target[target_arr == 2] = 1
        remapped_target[target_arr == 3] = 1
        
        target_img = Image.fromarray(remapped_target)
        
        # Basic transformations: Resize and to Tensor
        pre_img = TF.resize(pre_img, (self.img_size, self.img_size))
        post_img = TF.resize(post_img, (self.img_size, self.img_size))
        target_img = TF.resize(target_img, (self.img_size, self.img_size), interpolation=Image.NEAREST)
        
        pre_tensor = TF.to_tensor(pre_img) # 3 x H x W
        post_tensor = TF.to_tensor(post_img) # 1 x H x W
        target_tensor = torch.from_numpy(np.array(target_img)).long() # H x W
        
        # Concatenate pre and post along channels
        input_tensor = torch.cat([pre_tensor, post_tensor], dim=0) # 4 x H x W
        
        return input_tensor, target_tensor
