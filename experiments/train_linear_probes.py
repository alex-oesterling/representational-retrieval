import sys
sys.path.append(sys.path[0] + "/..")
from representational_retrieval import *
import torch
import numpy as np
from tqdm.auto import tqdm
import argparse
import clip
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import glob

## grab dataset clip embeddings

## train on fairface to classify race and age

## evaluate and save on top 10k for all 10 queries for CelebA, UTKFace, and occupations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default="cuda", type=str)
    parser.add_argument('-dataset', default="celeba", type=str)
    args = parser.parse_args()
    print(args)

    # _, preprocess = clip.load("ViT-B/32", device=args.device)
    
    # traindataset = FairFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", train=True, transform=preprocess, embedding_model="clip", binarize_age=True)
    # testdataset = FairFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", train=True, transform=preprocess, embedding_model="clip", binarize_age=True)
    # train_loader = torch.utils.data.DataLoader(traindataset, batch_size=512, shuffle=True)

    # fairface_embeds = []
    # fairface_age_labels = []
    # fairface_race_labels = []

    # for embed, labels in tqdm(train_loader, desc="Train"):
    #     fairface_embeds.append(embed.cpu())
    #     fairface_age_labels.append(labels[:, 1])
    #     fairface_race_labels.append(torch.argmax(labels[:, 2:], dim=-1))
    # if fairface_embeds[-1].shape[0] != 512:  
    #     fairface_embeds_temp = torch.stack(fairface_embeds[:-1]).cpu().flatten(0,1)
    #     fairface_age_labels_temp = torch.stack(fairface_age_labels[:-1]).cpu().flatten(0,1)
    #     fairface_race_labels_temp = torch.stack(fairface_race_labels[:-1]).cpu().flatten(0,1)
    # fairface_embeds = torch.cat((fairface_embeds_temp, fairface_embeds[-1].cpu()), dim=0).numpy()
    # fairface_age_labels = torch.cat((fairface_age_labels_temp, fairface_age_labels[-1].cpu()), dim=0).numpy()
    # fairface_race_labels = torch.cat((fairface_race_labels_temp, fairface_race_labels[-1].cpu()), dim=0).numpy()

    # parameters = {'C':[0.01, 0.1, 1, 10, 100]}

    # print(fairface_embeds.shape)
    # print(fairface_age_labels.shape)
    # print(fairface_race_labels.shape)

    # lr_age = LogisticRegression(penalty="l2", C=1)
    # lr_race = LogisticRegression(C=1, multi_class="multinomial", solver="saga")

    # clf_age = GridSearchCV(lr_age, parameters)
    # clf_race = GridSearchCV(lr_race, parameters)

    # clf_age.fit(fairface_embeds, fairface_age_labels)
    # clf_race.fit(fairface_embeds, fairface_race_labels)

    # print("Train age:", clf_age.score(fairface_embeds, fairface_age_labels))
    # print("Train race:", clf_race.score(fairface_embeds, fairface_race_labels))

    # test_loader = torch.utils.data.DataLoader(testdataset, batch_size=512, shuffle=False)

    # fairface_embeds = []
    # fairface_age_labels = []
    # fairface_race_labels = []
    # for embed, labels in tqdm(test_loader, desc="Test"):
    #     fairface_embeds.append(embed.cpu())
    #     fairface_age_labels.append(labels[:, 1])
    #     fairface_race_labels.append(torch.argmax(labels[:, 2:], dim=-1))
    # if fairface_embeds[-1].shape[0] != 512:  
    #     fairface_embeds_temp = torch.stack(fairface_embeds[:-1]).cpu().flatten(0,1)
    #     fairface_age_labels_temp = torch.stack(fairface_age_labels[:-1]).cpu().flatten(0,1)
    #     fairface_race_labels_temp = torch.stack(fairface_race_labels[:-1]).cpu().flatten(0,1)
    # fairface_embeds = torch.cat((fairface_embeds_temp, fairface_embeds[-1].cpu()), dim=0).numpy()
    # fairface_age_labels = torch.cat((fairface_age_labels_temp, fairface_age_labels[-1].cpu()), dim=0).numpy()
    # fairface_race_labels = torch.cat((fairface_race_labels_temp, fairface_race_labels[-1].cpu()), dim=0).numpy()

    # print("Test age:",clf_age.score(fairface_embeds, fairface_age_labels))
    # print("Test race:", clf_race.score(fairface_embeds, fairface_race_labels))

    # ## embed new dataset:
    # dataset = CelebA("/n/holylabs/LABS/calmon_lab/Lab/datasets/", attributes=None, train=True, transform=preprocess, embedding_model="clip")

    for embedpath in glob.glob("/n/holylabs/LABS/calmon_lab/Lab/datasets/celeba/clip/*/probe_labels.npy"):
        probe_labels = np.load(embedpath)
        print(probe_labels[0])
        probe_labels = np.delete(probe_labels, 1, axis=1)
        print(probe_labels[0])
        print(probe_labels.shape)
        np.save(os.path.join(os.path.split(embedpath)[0], "probe_labels.npy"), probe_labels)

    for embedpath in glob.glob("/n/holylabs/LABS/calmon_lab/Lab/datasets/occupations/clip/*/probe_labels.npy"):
        probe_labels = np.load(embedpath)
        print(probe_labels[0])
        probe_labels = np.delete(probe_labels, 1, axis=1)
        print(probe_labels[0])
        print(probe_labels.shape)
        np.save(os.path.join(os.path.split(embedpath)[0], "probe_labels.npy"), probe_labels)

    # dataset = Occupations("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, embedding_model="clip")

    # for embedpath in glob.glob("/n/holylabs/LABS/calmon_lab/Lab/datasets/occupations/clip/architect/embeds.npy"):
    #     embed = np.load(embedpath)
    #     agelabels = clf_age.predict_proba(embed)
    #     racelabels = clf_race.predict_proba(embed)
    #     genderlabels = []
    #     with open(os.path.join(os.path.split(embedpath)[0], "images.txt"), "r") as f:
    #         lines = f.readlines()
    #         for i, line in enumerate(lines):
    #             line = line.strip()
    #             idx = dataset.img_paths.index(line+".jpg")
    #             genderlabels.append(dataset.labels[idx][0].numpy())
    #     genderlabels = np.stack(genderlabels).reshape(-1,1)
    #     labels_full = np.concatenate((genderlabels, agelabels, racelabels), axis=1)
    #     np.save(os.path.join(os.path.split(embedpath)[0], "probe_labels.npy"), labels_full)

    # exit()
    # dataset = FairFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, embedding_model="clip", binarize_age=True)

    # for embedpath in glob.glob("/n/holylabs/LABS/calmon_lab/Lab/datasets/fairface/clip/*/embeds.npy"):
    #     # embed = np.load(embedpath)
    #     agelabels = []
    #     racelabels = []
    #     genderlabels = []
    #     with open(os.path.join(os.path.split(embedpath)[0], "images.txt"), "r") as f:
    #         lines = f.readlines()
    #         for i, line in enumerate(lines):
    #             line = line.strip()
    #             idx = dataset.img_paths.index(line+".jpg")
    #             genderlabels.append(dataset.labels[idx][0].numpy())
    #             racelabels.append(dataset.labels[idx][2:].numpy())
    #             agelabels.append(dataset.labels[idx][1].numpy())
    #     genderlabels = np.stack(genderlabels).reshape(len(lines), -1)
    #     racelabels = np.stack(racelabels).reshape(len(lines), -1)
    #     agelabels = np.stack(agelabels).reshape(len(lines), -1)

    #     labels_full = np.concatenate((genderlabels, agelabels, racelabels), axis=1)
    #     print(labels_full.shape)
    #     np.save(os.path.join(os.path.split(embedpath)[0], "probe_labels.npy"), labels_full)
 
    # pred_loader = 

    





    

if __name__ == "__main__":
    main()