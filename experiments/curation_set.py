from representational_retrieval import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse
import pickle
import seaborn as sns
import clip
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader
sns.set_style("whitegrid")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default="cuda", type=str)
    parser.add_argument('-dataset', default="celeba", type=str)
    parser.add_argument('-curation_set', default="fairface", type=str)
    parser.add_argument('-n_samples', default=10000, type=int)
    parser.add_argument('-query', default="A photo of a CEO", type=str)
    parser.add_argument('-k', default=10, type=int)
    parser.add_argument('-functionclass', default="linearregression", type=str)
    args = parser.parse_args()

    model, preprocess = clip.load("ViT-B/32", device=args.device)

    # Load the dataset
    if args.dataset == "fairface":
        dataset = FairFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", train=True, transform=preprocess)
    elif args.dataset == "occupations":
        pass
    elif args.dataset == "celeba":
        dataset = CelebA("/n/holylabs/LABS/calmon_lab/Lab/datasets/", attributes=None, train=True, transform=preprocess)
    else:
        print("Dataset not supported!")
        exit()

    if args.functionclass == "randomforest":
        oracle = RandomForestRegressor(max_depth=2)
    elif args.functionclass == "linearregression":
        oracle = LinearRegression()
    else:
        print("Function class not supported.")
        exit()

    batch_size = 512

    all_features = []
    all_labels = []
    ix = 0
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=batch_size)):
            features = model.encode_image(images.to(args.device))

            all_features.append(features)
            all_labels.append(labels)

            ix += batch_size
            if ix>=args.n_samples:
                break

    features, labels = torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
    m = labels.shape[0]

    q = args.query

    q_token = clip.tokenize(q).to(args.device) 
    with torch.no_grad():
        q_emb = model.encode_text(q_token).cpu().numpy().astype(np.float64)
    q_emb = q_emb/np.linalg.norm(q_emb)

    features = features.astype(np.float64)
    features = features/ np.linalg.norm(features, axis=1, keepdims=True)  # Calculate L2 norm for each row