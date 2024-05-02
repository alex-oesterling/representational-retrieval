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
    parser.add_argument('-retrieval_set', default="celeba", type=str)
    parser.add_argument('-curation_set', default="fairface", type=str)
    parser.add_argument('-retrieval_set_size', default=10000, type=int)
    parser.add_argument('-curation_set_size', default=10000, type=int)
    parser.add_argument('-query', default="A photo of a CEO", type=str)
    parser.add_argument('-k', default=10, type=int)
    parser.add_argument('-functionclass', default="linearregression", type=str)
    args = parser.parse_args()

    model, preprocess = clip.load("ViT-B/32", device=args.device)

    # Load the dataset
    if args.retrieval_set == "fairface":
        dataset = FairFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", train=True, transform=preprocess)
    elif args.retrieval_set == "occupations":
        dataset = Occupations("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess)
    elif args.retrieval_set == "utkface":
        dataset = UTKFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess)
    elif args.retrieval_set == "celeba":
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

    retrieval_features = []
    retrieval_labels = []
    ix = 0
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=batch_size)):
            features = model.encode_image(images.to(args.device))

            retrieval_features.append(features)
            retrieval_labels.append(labels[:, 21].unsqueeze(1))

            ix += batch_size
            if ix>=args.retrieval_set_size:
                break

    retrieval_features, retrieval_labels = torch.cat(retrieval_features).cpu().numpy(), torch.cat(retrieval_labels).cpu().numpy()
    n = retrieval_labels.shape[0]

    q = args.query

    q_token = clip.tokenize(q).to(args.device) 
    with torch.no_grad():
        q_emb = model.encode_text(q_token).cpu().numpy().astype(np.float64)
    q_emb = q_emb/np.linalg.norm(q_emb)

    retrieval_features = retrieval_features.astype(np.float64)
    retrieval_features = retrieval_features/ np.linalg.norm(retrieval_features, axis=1, keepdims=True)  # Calculate L2 norm for each row

    if args.curation_set == "fairface":
        curation_set = FairFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", train=True, transform=preprocess)
    elif args.curation_set == "occupations":
        curation_set = Occupations("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess)
    elif args.curation_set == "utkface":
        curation_set = UTKFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess)
    elif args.curation_set == "celeba":
        curation_set = CelebA("/n/holylabs/LABS/calmon_lab/Lab/datasets/", attributes=None, train=True, transform=preprocess)
    else:
        print("Curation set not supported!")
        exit()

    batch_size = 512

    curated_features = []
    curated_labels = []
    ix = 0
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(curation_set, batch_size=batch_size)):
            feat = model.encode_image(images.to(args.device))

            curated_features.append(feat)
            curated_labels.append(labels[:, 0].unsqueeze(1))

            ix += batch_size
            if ix>=args.curation_set_size:
                break

    curation_features, curation_labels = torch.cat(curated_features).cpu().numpy(), torch.cat(curated_labels).cpu().numpy()
    m = curation_labels.shape[0]

    curation_features = curation_features.astype(np.float64)
    curation_features = curation_features/ np.linalg.norm(curation_features, axis=1, keepdims=True)  # Calculate L2 norm for each row

    s = retrieval_features @ q_emb.T

    top_indices = np.zeros(n)
    top_indices[np.argsort(s.squeeze())[::-1][:args.k]] = 1
    sim_upper_bound = s.T@top_indices
    print("Similarity Upper Bound", sim_upper_bound, flush=True)

    solver = GreedyOracle(s, dataset=retrieval_labels, curation_set=curation_labels, model = oracle)

    reps_final = []
    sims_final = []
    rounded_reps_final = []
    rounded_sims_final = []
    sparsities = []
    rounded_indices_list = []
    relaxed_indices_list = []
    rhos = np.logspace(-4, 0.5, 20)
    for rho in rhos:
        indices = solver.fit(args.k, 10, rho)
        sparsity = sum(indices>1e-4)
        indices_rounded = indices.copy()
        indices_rounded[np.argsort(indices_rounded)[::-1][args.k:]] = 0
        indices_rounded[indices_rounded>1e-5] = 1.0 

        print(np.sort(indices)[::-1], flush=True)
        print(np.sort(indices_rounded)[::-1], flush=True)
        print(np.linalg.norm(indices-indices_rounded), flush=True)

        rep = solver.get_representation(indices, args.k)
        sim = solver.get_similarity(indices)

        rounded_rep = solver.get_representation(indices_rounded, args.k)
        rounded_sim = solver.get_similarity(indices_rounded)

        reps_final.append(rep)
        sims_final.append(sim)
        rounded_reps_final.append(rounded_rep)
        rounded_sims_final.append(rounded_sim)
        sparsities.append(sparsity)
        rounded_indices_list.append(indices_rounded)
        relaxed_indices_list.append(indices)

    rep_upper_bound = solver.get_representation(top_indices, args.k)

    print("Sim upper bound:", sim_upper_bound)
    print("Rep lower bound:", rep_upper_bound)

    results = {}
    results['relaxed_MPR'] = reps_final
    results['relaxed_sims'] = sims_final
    results['rounded_MPR'] = rounded_reps_final
    results['rounded_sims'] = rounded_sims_final
    results['relaxed_indices'] = relaxed_indices_list
    results['rounded_indices'] = rounded_indices_list
    results['rhos'] = rhos
    result_path = './results/alex/'
    filename_pkl = "{}_curation_oracle_{}_{}.pkl".format(args.retrieval_set, args.functionclass, args.k)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path + filename_pkl, 'wb') as f:
        pickle.dump(results, f)

    plt.figure()
    plt.title("Curation+Oracle, {}, retrieval {}, curation {}".format(args.functionclass, args.retrieval_set, args.curation_set))
    plt.plot(rounded_reps_final, rounded_sims_final, label="Binary")
    plt.plot(reps_final,sims_final, label="Relaxed")
    plt.axhline(y=sim_upper_bound, color='b', linestyle=':', label="Similarity Upper Bound")
    plt.axvline(x=rep_upper_bound, color='r', linestyle=':', label="Representation Lower Bound")
    plt.xlabel('Representation')
    plt.ylabel('Similarity')
    plt.legend()
    plt.savefig("./results/{}_curation_oracle_{}_{}_rep_sim.png".format(args.functionclass, args.retrieval_set, args.curation_set))

    plt.figure()
    plt.title("Curation+Oracle, {}, retrieval {}, curation {}".format(args.functionclass, args.retrieval_set, args.curation_set))
    plt.plot(reps_final, sparsities)
    plt.xlabel('Representation')
    plt.ylabel('Sparsity')
    plt.savefig("./results/{}_curation_oracle_{}_{}_rep_spar.png".format(args.functionclass, args.retrieval_set, args.curation_set))

if __name__ == "__main__":
    main()