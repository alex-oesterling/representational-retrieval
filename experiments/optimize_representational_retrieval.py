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

    # compute similarities
    s = features @ q_emb.T

    top_indices = np.zeros(m)
    top_indices[np.argsort(s.squeeze())[::-1][:args.k]] = 1
    sim_upper_bound = s.T@top_indices
    print("Similarity Upper Bound", sim_upper_bound, flush=True)

    solver = GreedyOracle(s, labels, model = oracle)

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
    result_path = './results/'
    filename_pkl = "{}_oracle_{}_{}.pkl".format(args.dataset, args.k, args.functionclass)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path + filename_pkl, 'wb') as f:
        pickle.dump(results, f)

    plt.figure()
    plt.title("Oracle, {}, {}".format(args.functionclass, args.dataset))
    plt.plot(rounded_reps_final, rounded_sims_final, label="Binary")
    plt.plot(reps_final,sims_final, label="Relaxed")
    plt.axhline(y=sim_upper_bound, color='b', linestyle=':', label="Similarity Upper Bound")
    plt.axvline(x=rep_upper_bound, color='r', linestyle=':', label="Representation Lower Bound")
    plt.xlabel('Representation')
    plt.ylabel('Similarity')
    plt.legend()
    plt.savefig("./results/greedy_{}_{}_rep_sim.png".format(args.functionclass, args.dataset))

    plt.figure()
    plt.title("Oracle, {}, {}".format(args.functionclass, args.dataset))
    plt.plot(reps_final, sparsities)
    plt.xlabel('Representation')
    plt.ylabel('Sparsity')
    plt.savefig("./results/greedy_{}_{}_rep_spar.png".format(args.functionclass, args.dataset))


        # solver = CVXPY(s, labels)
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
    #     rep = solver.get_representation(indices, args.k)
    #     sim = solver.get_similarity(indices)

    #     sparsity = sum(indices>1e-4)
    #     indices_rounded = indices
    #     indices_rounded[np.argsort(indices_rounded)[::-1][args.k:]] = 0
    #     indices_rounded[indices_rounded>1e-5] = 1.0 

    #     rounded_rep = solver.get_representation(indices_rounded, args.k)
    #     rounded_sim = solver.get_similarity(indices_rounded)

    #     reps.append(rounded_rep)
    #     sims.append(rounded_sim)
    #     relaxed_reps.append(rep)
    #     relaxed_sims.append(sim)
    #     spars.append(sparsity)

    # plt.figure()
    # plt.plot(reps, sims, label="Binary")
    # plt.plot(relaxed_reps,relaxed_sims, label="Relaxed")
    # plt.axhline(y=sim_upper_bound, color='b', linestyle=':')
    # plt.xlabel('Representation')
    # plt.ylabel('Similarity')
    # plt.legend()
    # plt.savefig("./results/linear_rep_sim.png")

    # plt.figure()
    # plt.plot(reps, spars)
    # plt.xlabel('Representation')
    # plt.ylabel('Sparsity')
    # plt.savefig("./results/linear_rep_spar.png")
if __name__ == "__main__":
    main()