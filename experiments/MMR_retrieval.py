from representational_retrieval import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse
import seaborn as sns
import clip
# from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
sns.set_style("whitegrid")
import pickle
import os

def mmr():
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default="cuda", type=str)
    parser.add_argument('-n_samples', default=10000, type=int)
    parser.add_argument('-k', default=10, type=int)
    args = parser.parse_args()

    model, preprocess = clip.load("ViT-B/32", device=args.device)

    # Load the dataset
    dataset = CelebA("/n/calmon_lab/Lab/datasets/", attributes=None, train=True, transform=preprocess)

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

    q = 'A photo of a Professor.'

    q_token = clip.tokenize(q).to(args.device)
    with torch.no_grad():
        q_emb = model.encode_text(q_token).cpu().numpy().astype(np.float64)
    q_emb = q_emb/np.linalg.norm(q_emb)

    features = features.astype(np.float64)
    features = features/ np.linalg.norm(features, axis=1, keepdims=True)  # Calculate L2 norm for each row

    # compute similarities
    s = features @ q_emb.T

    solver = MMR(s, labels, features)
    pvals = np.linspace(1e-5, 1-1e-5, 50)

    reps = []
    sims = []
    indices_list = []
    selection_list = []
    MMR_cost_list = []
    for p in tqdm(pvals):
        indices, diversity_cost, selection = solver.fit(args.k, p) 
        # c(x)
        weighting_clf = oracle_function(indices, labels, model=LinearRegression)
        # c(x) evaluated on data points and normalized
        weighting_vector  = weighting_clf.predict(labels)
        weighting_vector/= np.linalg.norm(weighting_vector)
        rep = np.sum((1/args.k)*indices* weighting_vector -(1/m)*weighting_vector) # weighting_vector is c
        sim = (s.T @ indices)
        reps.append(rep)
        sims.append(sim)
        indices_list.append(indices)
        selection_list.append(selection)
        MMR_cost_list.append(diversity_cost)

    results = {}
    results['MPR'] = reps
    results['sims'] = sims
    results['indices'] = indices_list
    results['pvals'] = pvals
    results['selection'] = selection_list
    results['MMR_cost'] = MMR_cost_list
    result_path = '../results/'
    filename_pkl = "mmr_rep_sim.pkl"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path + filename_pkl, 'wb') as f:
        pickle.dump(results, f)
    
    plt.figure()
    plt.plot(reps, sims, label="Binary")
    plt.xlabel('Representation')
    plt.ylabel('Similarity')
    plt.savefig("../results/mmr_rep_sim.png")
if __name__ == "__main__":
    main()