import sys
sys.path.append(sys.path[0] + "/..")
from representational_retrieval import *
import torch
import numpy as np
from tqdm.auto import tqdm
import argparse
import seaborn as sns
import clip
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
sns.set_style("whitegrid")
import pickle
import os
import debias_clip as dclip
import json

def argmax_probe_labels(probe_embeds):
    age_vec = (probe_embeds[:, 1] > 0.5).astype(np.int64)
    race_vec = probe_embeds[:, 2:]
    race_argmax = np.argmax(race_vec, axis=1)
    race_onehot = np.zeros((race_argmax.size, probe_embeds.shape[1]-2))
    race_onehot[np.arange(race_argmax.size), race_argmax] = 1
    return np.concatenate((probe_embeds[:, 0].reshape(-1, 1), age_vec.reshape(-1,1), race_onehot), axis=1)

def get_top_embeddings_labels_ids(dataset, query, embedding_model, datadir):
    if datadir == "occupations": ## Occupations is so small no need to use all 10k
        embeddings = []
        filepath = "/n/holylabs/LABS/calmon_lab/Lab/datasets/occupations/"
        embeddings = np.load(os.path.join(filepath, embedding_model, "architect/embeds.npy"))
        probe_labels = np.load(os.path.join(filepath, embedding_model, "architect/probe_labels.npy"))
        indices = []
        labels = np.zeros((embeddings.shape[0], dataset.labels.shape[1]))
        with open(os.path.join(filepath, embedding_model, "architect/images.txt"), "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                idx = dataset.img_paths.index(line+".jpg")
                labels[i] = dataset.labels[idx].numpy()
                indices.append(idx)
        # labels = dataset.labels.numpy()
        # indices = torch.arange(embeddings.shape[0])        
    else:
        retrievaldir = os.path.join("/n/holylabs/LABS/calmon_lab/Lab/datasets", datadir, embedding_model,query)
        embeddings = np.load(os.path.join(retrievaldir, "embeds.npy"))
        probe_labels = np.load(os.path.join(retrievaldir, "probe_labels.npy"))
        # embeddings /= np.linalg.norm(embeddings, axis=1)
        labels = np.zeros((embeddings.shape[0], dataset.labels.shape[1]))
        indices = []
        with open(os.path.join(retrievaldir, "images.txt"), "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                idx = dataset.img_paths.index(line+".jpg")
                labels[i] = dataset.labels[idx].numpy()
                indices.append(idx)
    
    return embeddings, labels, indices, probe_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default="cuda", type=str)
    parser.add_argument('-dataset', default="celeba", type=str)
    parser.add_argument('-curation_dataset', default=None, type=str)
    parser.add_argument('-query', default="queries.txt", type=str)
    parser.add_argument('-k', default=10, type=int)
    parser.add_argument('-functionclass', default="randomforest", type=str)
    parser.add_argument('-method', default="mmr", type=str)
    parser.add_argument('-use_clip', action="store_true")
    args = parser.parse_args()
    print(args)

    if args.method not in ['mmr', 'pbm', 'debiasclip', 'clipclip']:
        print("method not supported")
        exit()

    if args.method != "debiasclip":
        embedding_model = "clip"
        # model, preprocess = clip.load("ViT-B/32", device=args.device)
        _, preprocess = clip.load("ViT-B/32", device=args.device)
        # model = model.to(args.device)

    else:
        embedding_model = "debiasclip"
        _, preprocess = dclip.load("ViT-B/16-gender", device =args.device) # DebiasClip for gender, the only publicly available model
        # model = model.to(args.device)

    if args.functionclass == "randomforest":
        reg_model = RandomForestRegressor(max_depth=2, random_state=1017)
    elif args.functionclass == "linearregression":
        reg_model = LinearRegression(fit_intercept=False)
    elif args.functionclass == "decisiontree":
        reg_model = DecisionTreeRegressor(max_depth=3, random_state=1017)
    elif args.functionclass == "mlp":
        reg_model = MLPRegressor([64, 64], random_state=1017)
    else:
        print("Function class not supported.")
        exit()

    if args.dataset == "fairface":
        dataset = FairFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", train=True, transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "occupations":
        dataset = Occupations("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "utkface":
        dataset = UTKFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "celeba":
        dataset = CelebA("/n/holylabs/LABS/calmon_lab/Lab/datasets/", attributes=None, train=True, transform=preprocess, embedding_model=embedding_model)
        
    if args.curation_dataset:
        if args.curation_dataset == "fairface":
            curation_dataset = FairFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", train=True, transform=preprocess, embedding_model=embedding_model)
        elif args.curation_dataset == "occupations":
            curation_dataset = Occupations("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, embedding_model=embedding_model)
        elif args.curation_dataset == "utkface":
            curation_dataset = UTKFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, embedding_model=embedding_model)
        elif args.curation_dataset == "celeba":
            curation_dataset = CelebA("/n/holylabs/LABS/calmon_lab/Lab/datasets/", attributes=None, train=True, transform=preprocess, embedding_model=embedding_model)
        else:
            print("Curation set not supported!")
            exit()
    else:
        curation_dataset = None

    with open(args.query, 'r') as f:
        queries = f.readlines()

    for query in tqdm(queries[::-1]):
        q_org = query.strip()
        q = "A photo of "+ q_org
        q_tag = " ".join(q.split(" ")[4:])
        print(q_tag)

        path = "/n/holyscratch01/calmon_lab/Users/aoesterling/representational_retrieval/final_results/"
        with open(path+f'{args.dataset}_curation_fairface_top10k_{args.method}_{args.k}_linearregression_{q_tag}.pkl', 'rb') as f:
            results_mmr = pickle.load(f)

        retrieval_features, retrieval_labels, retrieval_indices, retrieval_probe_labels = get_top_embeddings_labels_ids(
            dataset,
            q_tag,
            embedding_model,
            args.dataset
        )
        print(retrieval_probe_labels.shape)

        retrieval_features = retrieval_features.astype(np.float32)

        retrieval_probe_labels = argmax_probe_labels(retrieval_probe_labels)
        
        if curation_dataset is not None:
            curation_features, curation_labels, curation_indices, curation_probe_labels = get_top_embeddings_labels_ids(
                curation_dataset,
                q_tag,
                embedding_model,
                args.curation_dataset
            )
            curation_features = curation_features.astype(np.float32)
            curation_probe_labels = argmax_probe_labels(curation_probe_labels)
            curation_labels_full = curation_labels

            if args.dataset == "utkface": ## remap races to utkface
                new_races = np.zeros((curation_probe_labels.shape[0], 5))
                new_races[:, 0] = curation_probe_labels[:, 5]
                new_races[:, 1] = curation_probe_labels[:, 4]
                new_races[:, 2] = np.logical_or(curation_probe_labels[:, 2], curation_probe_labels[:, 8]) ## Asian, SE Asian
                new_races[:, 3] = curation_probe_labels[:, 3]
                new_races[:, 4] = np.logical_or(curation_probe_labels[:, 6], curation_probe_labels[:, 7]) ## Middle Eastern, Latino
                curation_probe_labels = np.concatenate((curation_probe_labels[:, :2], new_races), axis=1)
            # if args.curation_dataset == "utkface":
            #     new_races = np.zeros((retrieval_probe_labels.shape[0], 5))
            #     new_races[:, 0] = retrieval_probe_labels[:, 5]
            #     new_races[:, 1] = retrieval_probe_labels[:, 4]
            #     new_races[:, 2] = np.logical_or(retrieval_probe_labels[:, 2], retrieval_probe_labels[:, 8]) ## Asian, SE Asian
            #     new_races[:, 3] = retrieval_probe_labels[:, 3]
            #     new_races[:, 4] = np.logical_or(retrieval_probe_labels[:, 6], retrieval_probe_labels[:, 7]) ## Middle Eastern, Latino
            #     retrieval_probe_labels = np.concatenate((retrieval_probe_labels[:, :2], new_races), axis=1)

            # synthetic_curation_probe_labels = None
            # num_races = curation_probe_labels.shape[1]-2
            # for i in range(2):
            #     for j in range(2):
            #         temp = np.concatenate([np.ones((num_races, 1))*i,np.ones((num_races, 1))*j,np.eye(num_races)], axis=1)
            #         if synthetic_curation_probe_labels is None:
            #             synthetic_curation_probe_labels = temp
            #         else:
            #             synthetic_curation_probe_labels = np.concatenate([synthetic_curation_probe_labels, temp], axis=0)
            # print(synthetic_curation_probe_labels.shape)

            # curation_labels = synthetic_curation_probe_labels
            curation_labels = curation_probe_labels
            retrieval_labels = retrieval_probe_labels

            print(curation_labels.shape)
            print(retrieval_labels.shape)
            print(retrieval_labels[0])
        else:
            curation_features = None
            curation_labels = None
            curation_labels_full = None

        new_results = {}
        new_results['sims'] = results_mmr['sims']
        new_results['indices'] = results_mmr['indices']
        
        mprs = []

        if args.method == "debiasclip":
            mpr, _ = getMPR(new_results["indices"], retrieval_labels, args.k, curation_set = curation_labels, model=reg_model)
            mprs = mpr
        else:
            for index in new_results["indices"]:
                mpr, _ = getMPR(index, retrieval_labels, args.k, curation_set = curation_labels, model=reg_model)
                mprs.append(mpr)

        new_results['MPR'] = mprs

        result_path = '/n/holyscratch01/calmon_lab/Users/aoesterling/representational_retrieval/final_results/'
        q_title = q.split(" ")[-1]
        print("MPR: ", new_results['MPR'])
        print("sims: ", new_results['sims'])

        if args.use_clip:
            filename_pkl = "clip_{}_curation_{}_top10k_{}_{}_{}_{}.pkl".format(args.dataset, args.curation_dataset, args.method, args.k, args.functionclass, q_title)
        else:
            filename_pkl = "{}_curation_{}_top10k_{}_{}_{}_{}.pkl".format(args.dataset, args.curation_dataset, args.method, args.k, args.functionclass, q_title)
        # filename_pkl = "{}_curation_{}_top10k_{}_{}_{}_{}.pkl".format(args.dataset, args.curation_dataset, args.method, args.k, args.functionclass, q_title)
        print(filename_pkl)
        print(result_path+filename_pkl)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if os.path.exists(os.path.join(result_path,filename_pkl)):
            print("path exists", os.path.join(result_path,filename_pkl))
            exit()
        with open(os.path.join(result_path,filename_pkl), 'wb') as f:
            pickle.dump(new_results, f)

if __name__ == "__main__":
    main()