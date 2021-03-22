import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Crop(px=(1, 16), keep_size=True),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.25,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

customTransforms = ImgAugTransform()

class MyDataset(Dataset):

    def __init__(self, is_train=True, num_cat=200):
        self.num_cat = num_cat
        self.is_train = is_train
        self.data_path = "train"
        self._load_file_list()

    def __len__(self):
        return len(self.labels)

    def _load_file_list(self):
        image_files = []
        labels = []

        work_folders = [int(f) for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f)) and '.svn' not in f]
        work_folders.sort()
        work_folders = work_folders[:self.num_cat]
        for cat in work_folders:
            cat = str(cat)
            cat_folder = os.path.join(self.data_path, cat)
            images = [f for f in os.listdir(cat_folder) if os.path.isfile(os.path.join(cat_folder, f)) and '.svn' not in f]
            images = images[:-50] if self.is_train else images[-50:]

            for img in images:
                img_file = os.path.join(cat_folder, img)
                image_files.append(img_file)
                labels.append(cat)

        self.image_files = image_files
        self.labels = labels

    def __getitem__(self, idx):
        img = cv2.imread(self.image_files[idx])
        if img is None:
            return self.__getitem__((idx + 1) % self.__len__())

        img = Image.open(self.image_files[idx])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if self.is_train:
            # img_transforms = transforms.Compose([
            #     transforms.ColorJitter(hue=.05, saturation=.05),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomVerticalFlip(),
            #     transforms.RandomRotation(20),
            #     transforms.RandomErasing(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            img_transforms = transforms.Compose([
                customTransforms,
                transforms.ToTensor(),
                normalize
            ])
        else:
            img_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        img = img_transforms(img)
        label = int(self.labels[idx])

        sample = {
            'img': img,
            'label': label
        }

        return sample

class MyDatasetForFinetune(Dataset):

    def __init__(self, is_train=True, num_cat=200):
        self.num_cat = num_cat
        self.is_train = is_train
        self.data_path = "train"
        self._load_file_list()

    def __len__(self):
        return len(self.labels)

    def _load_file_list(self):
        image_files = []
        labels = []

        work_folders = [int(f) for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f)) and '.svn' not in f]
        work_folders.sort()
        work_folders = work_folders[:self.num_cat]
        for cat in work_folders:
            cat = str(cat)
            cat_folder = os.path.join(self.data_path, cat)
            images = [f for f in os.listdir(cat_folder) if os.path.isfile(os.path.join(cat_folder, f)) and '.svn' not in f]
            images = images[:-50] if self.is_train else images[-50:]

            for img in images:
                img_file = os.path.join(cat_folder, img)
                image_files.append(img_file)
                labels.append(cat)

        self.image_files = image_files
        self.labels = labels

    def __getitem__(self, idx):
        img = cv2.imread(self.image_files[idx])
        if img is None:
            return self.__getitem__((idx + 1) % self.__len__())

        img = Image.open(self.image_files[idx])
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        if self.is_train:
            img_transforms = transforms.Compose([
                transforms.Resize(size=(331, 331)),
                customTransforms,
                transforms.ToTensor(),
                normalize
            ])
        else:
            img_transforms = transforms.Compose([
                transforms.Resize(size=(331, 331)),
                transforms.ToTensor(),
                normalize
            ])
        img = img_transforms(img)
        label = int(self.labels[idx])

        sample = {
            'img': img,
            'label': label
        }

        return sample
