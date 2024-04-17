from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import torchvision
from PIL import Image
import os
import pandas as pd

class CelebA(torch.utils.data.Dataset):
    def __init__(self, path, attributes = None, train=True, transform=torchvision.transforms.ToTensor()):
        self.filepath = os.path.join(path, "celeba/")

        self.transform = transform
        self.img_paths = []
        self.labels = []

        self.attr_to_idx = {
            '5_o_Clock_Shadow': 1, 
            'Arched_Eyebrows': 2, 
            'Attractive': 3, 
            'Bags_Under_Eyes': 4, 
            'Bald': 5, 
            'Bangs': 6, 
            'Big_Lips': 7, 
            'Big_Nose': 8, 
            'Black_Hair': 9, 
            'Blond_Hair': 10, 
            'Blurry': 11, 
            'Brown_Hair': 12, 
            'Bushy_Eyebrows': 13, 
            'Chubby': 14, 
            'Double_Chin': 15, 
            'Eyeglasses': 16, 
            'Goatee': 17, 
            'Gray_Hair': 18, 
            'Heavy_Makeup': 19, 
            'High_Cheekbones': 20, 
            'Male': 21, 
            'Mouth_Slightly_Open': 22, 
            'Mustache': 23, 
            'Narrow_Eyes': 24, 
            'No_Beard': 25, 
            'Oval_Face': 26, 
            'Pale_Skin': 27, 
            'Pointy_Nose': 28, 
            'Receding_Hairline': 29, 
            'Rosy_Cheeks': 30, 
            'Sideburns': 31, 
            'Smiling': 32, 
            'Straight_Hair': 33, 
            'Wavy_Hair': 34, 
            'Wearing_Earrings': 35, 
            'Wearing_Hat': 36, 
            'Wearing_Lipstick': 37, 
            'Wearing_Necklace': 38, 
            'Wearing_Necktie': 39, 
            'Young': 40, 
        }

        if attributes is None:
            attributes = self.attr_to_idx.keys()

        attr_indices = [self.attr_to_idx[attr] for attr in attributes]

        if train:
            target_idx = 0
        else:
            target_idx = 1

        target_filepaths = set()
        with open(path + "celeba/list_eval_partition.txt", "r") as f:
            split_lines = f.readlines()
            for line in split_lines[1:]:
                line = line.strip()
                line = line.split()
                if int(line[1]) == target_idx:
                    target_filepaths.add(line[0])

        with open(path + "celeba/list_attr_celeba.txt", "r") as f:
            lines = f.readlines()
            lines = lines[2:]
            self.len = len(lines)
            for idx, line in enumerate(lines):
                line = line.strip()
                line = line.split()

                if line[0] not in target_filepaths:
                    continue

                self.img_paths.append(line[0])
                self.labels.append(torch.tensor([int(line[idx]) for idx in attr_indices]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        path = self.filepath + "img_align_celeba/" + self.img_paths[idx]
        image = Image.open(path)

        label = self.labels[idx]

        return self.transform(image), label
    
class Occupations(torch.utils.data.Dataset):
    def __init__(self, path, transform=torchvision.transforms.ToTensor()):
        self.filepath = os.path.join(path, "occupations/")

        self.transform = transform
        self.images = []
        self.labels = []

        df = pd.read_csv(self.filepath + "gender_labelled_images.csv")
        self.labels = df.image_gender

        construct_path = lambda x, y: os.path.join(self.filepath, "google", str(x), str(y)+".jpg")
        img_paths = [construct_path(*x) for x in tuple(zip(df['search_term'], df['order']))]

        for path in img_paths:
            self.images.append(self.transform(Image.open(path)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
class FairFace(torch.utils.data.Dataset):
    def __init__(self, path, train=True, transform=torchvision.transforms.ToTensor()):
        self.filepath = os.path.join(path, "fairface/")

        self.transform = transform
        # self.images = []
        self.image_paths = []
        self.labels = []

        if train:
            df = pd.read_csv(self.filepath + "fairface_label_train.csv")
        else:
            df = pd.read_csv(self.filepath + "fairface_label_val.csv")

        self.labels = [list(x) for x in list(zip(df.gender, df.race, df.age))]

        construct_path = lambda x: os.path.join(self.filepath, x)
        self.img_paths = [construct_path(x) for x in df.file]

        ## For speed can preload data, but too large to fit on GPU in general
        # for path in self.img_paths:
        #     self.images.append(self.transform(Image.open(path)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.img_paths[idx])), self.labels[idx]
    
class UTKFace(torch.utils.data.Dataset):
    def __init__(self, path, train=True, transform=torchvision.transforms.ToTensor()):
        # self.filepath = os.path.join(path, "fairface/")

        # self.transform = transform
        # self.images = []
        # self.labels = []

        # if train:
        #     df = pd.read_csv(self.filepath + "fairface_label_train.csv")
        # else:
        #     df = pd.read_csv(self.filepath + "fairface_label_val.csv")

        # self.labels = [list(x) for x in list(zip(df.gender, df.race, df.age))]

        # construct_path = lambda x: os.path.join(self.filepath, x)
        # img_paths = [construct_path(x) for x in df.file]

        # for path in img_paths:
        #     self.images.append(self.transform(Image.open(path)))
        return

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]