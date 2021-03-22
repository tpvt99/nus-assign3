import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, is_train=True, num_cat=200):
        self.num_cat = num_cat
        self.is_train = is_train
        self.data_path = "train_phong"
        self._load_file_list()

    def __len__(self):
        return len(self.labels)

    def _load_file_list(self):
        image_files = []
        labels = []

        work_folders = [int(f) for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))]
        work_folders.sort()
        work_folders = work_folders[:self.num_cat]
        for cat in work_folders:
            cat = str(cat)
            cat_folder = os.path.join(self.data_path, cat)
            images = [f for f in os.listdir(cat_folder) if os.path.isfile(os.path.join(cat_folder, f))]
            images = images[:-2] if self.is_train else images[-2:]

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

        img_transforms = transforms.Compose([
            transforms.ToTensor(),
            #normalize,
            transforms.RandomRotation(degrees=90),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
        img = img_transforms(img)
        label = int(self.labels[idx])

        sample = {
            'img': img,
            'label': label
        }

        return sample

train_set = MyDataset(is_train=True, num_cat=2)
train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

fig = plt.figure(figsize=(15,15))

counts = 1
for i in range(5):
    for z in train_loader:
        plt.subplot(5, 5, counts)
        img = z['img'].numpy().reshape(240, 320, 3)*250
        plt.imshow(img.astype('int32'))
        counts+=1
plt.show()
