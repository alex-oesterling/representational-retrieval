from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import torchvision
from PIL import Image
import os
import pandas as pd
import glob

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

        ## Labels is [gender, occupation_one_hot]

        df = pd.read_csv(self.filepath + "gender_labelled_images.csv")
        print(df.search_term) ## FIXME

        occupation_to_idx = {}
        for i, race in enumerate(df.search_term.unique()):
            occupation_to_idx[race] = i

        gender_to_idx = {
            'man': 0,
            'woman': 1
        }

        occupation_one_hot = torch.nn.functional.one_hot(torch.tensor([occupation_to_idx[occ] for occ in df.search_term]))

        gender = torch.tensor([gender_to_idx[gen] for gen in df.image_gender])

        self.labels = torch.tensor(torch.hstack([gender.unsqueeze(1), occupation_one_hot]))

        construct_path = lambda x, y: os.path.join(self.filepath, "google", str(x), str(y)+".jpg")
        img_paths = [construct_path(*x) for x in tuple(zip(df['search_term'], df['order']))]

        for path in img_paths:
            self.images.append(self.transform(Image.open(path)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
class FairFace(torch.utils.data.Dataset):
    def __init__(self, path, train=True, transform=torchvision.transforms.ToTensor(), race_one_hot=True):
        self.filepath = os.path.join(path, "fairface/")

        self.transform = transform
        # self.images = []
        self.image_paths = []
        self.labels = []

        if train:
            df = pd.read_csv(self.filepath + "fairface_label_train.csv")
        else:
            df = pd.read_csv(self.filepath + "fairface_label_val.csv")

        race_to_idx = {}
        for i, race in enumerate(df.race.unique()):
            race_to_idx[race] = i

        gender_to_idx = {
            'Male': 0,
            'Female': 1
        }

        age_to_idx = {
            '0-2': 0,
            '3-9': 1,
            '10-19': 2,
            '20-29': 3,
            '30-39': 4,
            '40-49': 5,
            '50-59': 6,
            '60-69': 7,
            'more than 70': 8
        }

        one_hot = torch.nn.functional.one_hot(torch.tensor([race_to_idx[race] for race in df.race])).numpy()
        gender_idx = [gender_to_idx[gen] for gen in df.gender]
        age_idx = [age_to_idx[age] for age in df.age]

        ## labels is [gender_binary, age_categorical, race_one_hot]

        self.labels = []
        for i in range(len(gender_idx)):
            self.labels.append([gender_idx[i], age_idx[i]] + list(one_hot[i]))

        self.labels = torch.tensor(self.labels)

        construct_path = lambda x: os.path.join(self.filepath, x)
        self.img_paths = [construct_path(x) for x in df.file]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.img_paths[idx])), self.labels[idx]
    
class UTKFace(torch.utils.data.Dataset):
    def __init__(self, path, train=True, transform=torchvision.transforms.ToTensor()):
        self.filepath = os.path.join(path, "utkface/")

        self.transform = transform
        self.imagepaths = []
        self.labels = []

        ## labels is [gender, age, race]

        for path in glob.glob(os.path.join(self.filepath, "*/*.jpg")):

            attributes = path.split("/")[-1].split("_")
            # if len(attributes) < 3:
            #     continue
            # print(path.split("/")[-1])
            # print(attributes)
            try:
                self.labels.append([int(attributes[1]), int(attributes[0]), int(attributes[2])])
            except:
                continue
            self.imagepaths.append(path)
        
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.imagepaths[idx])), self.labels[idx]