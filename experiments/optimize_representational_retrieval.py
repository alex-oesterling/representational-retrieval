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
    dataset = CelebA("/n/holylabs/LABS/hlakkaraju_lab/Lab/datasets/", attributes=None, train=True, transform=preprocess)

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

    # solver = CVXPY(s, labels)

    # lower, upper = solver.get_lower_upper_bounds(args.k)

    # pvals = np.linspace(lower+1e-5, upper-1e-5, 50)

    # reps = []
    # sims = []
    # relaxed_reps = []
    # relaxed_sims = []
    # spars = []
    # for p in tqdm(pvals):
    #     indices = solver.fit(args.k, p)
    #     # rep, sim = solver.fit(args.k, p)
    #     rep = np.sqrt(np.sum(np.power((labels.T @ indices)/args.k - (labels.T@np.ones(m))/m, 2)))
    #     sim = (s.T @ indices)
    #     reps.append(rep)
    #     sims.append(sim)

    # for p in tqdm(pvals):
    #     indices, spar = solver.fit_select(args.k, p)
    #     rep = np.sqrt(np.sum(np.power((labels.T @ indices)/args.k - (labels.T@np.ones(m))/m, 2)))
    #     sim = (s.T @ indices)
    #     relaxed_reps.append(rep)
    #     relaxed_sims.append(sim)
    #     spars.append(spar)

    # plt.figure()
    # plt.plot(reps, sims, label="Binary")
    # plt.plot(relaxed_reps,relaxed_sims, label="Relaxed")
    # plt.xlabel('Representation')
    # plt.ylabel('Similarity')
    # plt.savefig("./results/rep_sim.png")

    # plt.figure()
    # plt.plot(reps, spars)
    # plt.xlabel('Representation')
    # plt.ylabel('Sparsity')
    # plt.savefig("./results/rep_spar.png")

    solver = GreedyOracle(s, labels)

    final, reps, sims = solver.fit(10, 100, 1e-6)

    plt.figure()
    plt.plot(reps, sims, label="Binary")
    plt.xlabel('Representation')
    plt.ylabel('Similarity')
    plt.savefig("./results/greedy_rep_sim.png")
if __name__ == "__main__":
    main()