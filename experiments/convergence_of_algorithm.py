import sys
sys.path.append(sys.path[0] + "/..")
from representational_retrieval import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import clip
from sklearn.ensemble import RandomForestRegressor
sns.set_style("whitegrid")


def get_top_embeddings_labels_ids(dataset, query, embedding_model, datadir):
    if datadir == "occupations": ## Occupations is so small no need to use all 10k
        embeddings = []
        filepath = "/n/holylabs/LABS/calmon_lab/Lab/datasets/occupations/"
        for i, path in enumerate(dataset.img_paths):
            image_id = os.path.relpath(path.split(".")[0], filepath)
            embeddingpath = os.path.join(filepath, embedding_model, image_id+".pt")
            embeddings.append(torch.load(embeddingpath))
        embeddings = torch.nn.functional.normalize(torch.stack(embeddings), dim=1).cpu().numpy()
        labels = dataset.labels.numpy()
        indices = torch.arange(embeddings.shape[0])
    else:
        retrievaldir = os.path.join("/n/holylabs/LABS/calmon_lab/Lab/datasets", datadir, embedding_model,query)
        embeddings = np.load(os.path.join(retrievaldir, "embeds.npy"))
        embeddings /= np.linalg.norm(embeddings, axis=1)
        labels = np.zeros((embeddings.shape[0], dataset.labels.shape[1]))
        indices = []
        with open(os.path.join(retrievaldir, "images.txt"), "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                idx = dataset.img_paths.index(line+".jpg")
                labels[i] = dataset.labels[idx].numpy()
                indices.append(idx)
    
    return embeddings, labels, indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default="cuda", type=str)
    parser.add_argument('-dataset', default="celeba", type=str)
    parser.add_argument('-n_samples', default=10000, type=int)
    parser.add_argument('-query', default="A photo of a doctor", type=str)
    parser.add_argument('-k', default=10, type=int)
    parser.add_argument('-functionclass', default="linearregression", type=str)
    args = parser.parse_args()

    model, preprocess = clip.load("ViT-B/32", device=args.device)

    # Load the dataset
    if args.dataset == "fairface":
        dataset = FairFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", train=True, transform=preprocess)
    elif args.dataset == "occupations":
        dataset = Occupations("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess)
    elif args.dataset == "utkface":
        dataset = UTKFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess)
    elif args.dataset == "celeba":
        dataset = CelebA("/n/holylabs/LABS/calmon_lab/Lab/datasets/", attributes=None, train=True, transform=preprocess)
    else:
        print("Dataset not supported!")
        exit()

    if args.functionclass == "randomforest":
        oracle = RandomForestRegressor(max_depth=2)
    elif args.functionclass == "linearregression":
        oracle = LinearRegression(fit_intercept=False)
    else:
        print("Function class not supported.")
        exit()

    q = args.query

    q_token = clip.tokenize(q).to(args.device) 
    with torch.no_grad():
        q_emb = model.encode_text(q_token).cpu().numpy().astype(np.float64)
    q_emb = q_emb/np.linalg.norm(q_emb)

    retrieval_features, retrieval_labels, retrieval_indices = get_top_embeddings_labels_ids(
        dataset,
        "a doctor",
        "clip",
        args.dataset
    )

    m = retrieval_labels.shape[0]

    # compute similarities
    s = retrieval_features @ q_emb.T

    top_indices = np.zeros(m)
    top_indices[np.argsort(s.squeeze())[::-1][:args.k]] = 1
    sim_upper_bound = s.T@top_indices
    print("Similarity Upper Bound", sim_upper_bound, flush=True)

    solver2 = GurobiLP(s, retrieval_labels, model = oracle)
    solver = GurobiIP(s, retrieval_labels, model = oracle)

    reps_gurobi = []
    sims_gurobi = []
    reps_relaxed = []
    sims_relaxed = []
    sparsities = []
    rounded_reps_final = []
    rounded_sims_final = []
    relaxed_indices_list = []
    rounded_indices_list = []
    gurobi_indices_list = []
    rhoslist = []
    rho = 0.01
    iters = [1,2,5,10,20,50]
    for num_iters in iters:
        indices = solver2.fit(args.k, num_iters, rho)
        if indices is None:
            break
        indices_rounded = indices.copy()
        indices_rounded[np.argsort(indices_rounded)[::-1][args.k:]] = 0
        indices_rounded[indices_rounded>1e-5] = 1.0 

        rep = solver2.get_representation(indices, args.k)
        sim = solver2.get_similarity(indices)

        rounded_rep = solver2.get_representation(indices_rounded, args.k)
        rounded_sim = solver2.get_similarity(indices_rounded)

        reps_relaxed.append(rep)
        sims_relaxed.append(sim)
        rounded_reps_final.append(rounded_rep)
        rounded_sims_final.append(rounded_sim)
        rounded_indices_list.append(indices_rounded)
        relaxed_indices_list.append(indices)

        indices_gurobi = solver.fit(args.k, num_iters, rho)
        if indices_gurobi is None:
            break
        
        rep = solver.get_representation(indices_gurobi, args.k)
        sim = solver.get_similarity(indices_gurobi)

        gurobi_indices_list.append(indices_gurobi)

        reps_gurobi.append(rep)
        sims_gurobi.append(sim)

    print("final mprs", reps_relaxed)
    print("final sims", sims_relaxed)

    rep_upper_bound = solver.get_representation(top_indices, args.k)

    print("Sim upper bound:", sim_upper_bound)
    print("Rep lower bound:", rep_upper_bound)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("{}, {}".format(args.functionclass, args.dataset))
    ax.plot(iters, rounded_reps_final, label="LP-Rounded")
    ax.plot(iters, reps_relaxed, label="LP")
    ax.plot(iters, reps_gurobi, label="IP")
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(iters, rounded_sims_final, label="LP-Rounded", linestyle=":")
    ax2.plot(iters, sims_relaxed, label="LP", linestyle=":")
    ax2.plot(iters, sims_gurobi, label="IP", linestyle=":")
    # plt.plot(rhos[::-1], sims_final, label="Rho Constraints")
    ax.axhline(y=rho, color='black', linestyle='--', label="Target Rho")
    # plt.axvline(x=rep_upper_bound, color='r', linestyle=':', label="Representation Lower Bound")
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Representation')
    ax2.set_ylabel('Similarity')
    # ax2.legend(loc=0)
    plt.savefig("./results/iterstest_{}_{}_k_{}.png".format(args.functionclass, args.dataset, args.k))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("{}, {}".format(args.functionclass, args.dataset))
    ax.plot(iters, rounded_reps_final, label="LP-Rounded")
    ax.plot(iters, reps_relaxed, label="LP")
    ax.plot(iters, reps_gurobi, label="IP")
    ax.legend(loc=0)
    # ax2 = ax.twinx()
    # ax2.plot(iters, rounded_sims_final, label="LP-Rounded", linestyle=":")
    # ax2.plot(iters, sims_relaxed, label="LP", linestyle=":")
    # ax2.plot(iters, sims_gurobi, label="IP", linestyle=":")
    # plt.plot(rhos[::-1], sims_final, label="Rho Constraints")
    ax.axhline(y=rho, color='black', linestyle='--', label="Target Rho")
    # plt.axvline(x=rep_upper_bound, color='r', linestyle=':', label="Representation Lower Bound")
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Representation')
    # ax2.set_ylabel('Similarity')
    # ax2.legend(loc=0)
    plt.savefig("./results/iterstest_nosims_{}_{}_k_{}.png".format(args.functionclass, args.dataset, args.k))

    # plt.figure()
    # plt.title("Oracle, {}, {}".format(args.functionclass, args.dataset))
    # plt.plot(reps_relaxed, sparsities)
    # plt.xlabel('Representation')
    # plt.ylabel('Sparsity')
    # plt.savefig("./results/gurobi_{}_{}_rep_spar.png".format(args.functionclass, args.dataset))

if __name__ == "__main__":
    main()