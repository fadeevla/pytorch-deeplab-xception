
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from os.path import join
import random
import numpy as np
import sys



seed = random.randrange(sys.maxsize)

class MFCDataset(Dataset):
    def _listdir(self, dirpath):
        print('dir_path=',dirpath)
        ll = (os.getcwd(), os.listdir())
        print('current_dir content/n',ll)

        for x in os.listdir(dirpath):
            if x.split('.')[-1] in self._imgexts:
                yield(x)
    def __init__(self, root, img_transform=None, mask_transform =None):
        self.root = root 
        self._imgexts = ['jpg','jpeg','png']
       
        self.datapoints = [os.path.splitext(x)[0] for x in self._listdir(join(root, "image"))]
        self.endings = [os.path.splitext(x)[1] for x in self._listdir(join(root, "image"))]
        self.NUM_CLASSES = len(self.datapoints)
        transforms = self.get_transforms()

        if img_transform is None:
            self.img_transform = transforms['img_transform']
        if mask_transform is None:
            self.mask_transform = transforms['mask_transform']

    def __len__(self):
        return len(self.datapoints)

    def get_transforms(self):
        img_transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
        #    transforms.RandomResizedCrop(size=0),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.5),
            transforms.RandomGrayscale(),
            transforms.ToTensor()
            ])

        mask_transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
        #    transforms.RandomResizedCrop(size=152,interpolation=PIL.Image.NEAREST),
            transforms.RandomRotation(180)
            ])
        return {'img_transform' : img_transform, 'mask_transform' : mask_transform}

    def __getitem__(self, idx):
        if isinstance(idx, int) or isinstance(idx, np.int64):
            datapoint = self.datapoints[idx]
            if self.endings[idx] == ".jpg":
                image = Image.open(join(self.root, "image",datapoint + ".jpg"))
                image = image.resize((152,152), Image.BILINEAR)
                mask = Image.open(join(self.root, "mask", datapoint + ".png"))
                mask = mask.resize((152,152), Image.BILINEAR)               

            if self.endings[idx] == ".jpeg":
                image = Image.open(join(self.root, "image",datapoint + ".jpeg"))
                image = image.resize((152,152), Image.BILINEAR)
                mask = Image.open(join(self.root, "mask", "2.png")).convert('RGB')
                mask = mask.resize((152,152), Image.BILINEAR)

            seed = random.randrange(sys.maxsize)
            if self.img_transform:
                random.seed(seed) 
                try:
                    image = self.img_transform(image)
                except:
                    print('exception idx=',idx)
            if self.mask_transform and self.endings[idx] == ".jpg": 
                random.seed(seed) 
                mask = self.mask_transform(mask)
                try:
                    mask = np.array(mask,  dtype=np.float32)[:,:,0]
                except:
                    print('idx=',idx,' datapoint=', datapoint)
            else:
                mask = np.array(mask,  dtype=np.float32)[:,:,0]
                mask = np.zeros_like(mask)
            return [image, mask]
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(idx, tuple) or isinstance(idx, list):
            return [self[i] for i in idx]
        else:
            raise ValueError(f'Slicing for idx is {type(idx)} is not implemented')