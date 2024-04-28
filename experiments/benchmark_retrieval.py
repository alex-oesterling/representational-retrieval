from representational_retrieval import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse
import seaborn as sns
import clip
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader
sns.set_style("whitegrid")
import pickle
import os
import debias_clip as dclip

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-method', default="mmr", type=str)
    parser.add_argument('-device', default="cuda", type=str)
    parser.add_argument('-dataset', default="celeba", type=str)
    parser.add_argument('-n_samples', default=10000, type=int)
    parser.add_argument('-query', default="A photo of a CEO", type=str)
    parser.add_argument('-k', default=10, type=int)
    parser.add_argument('-functionclass', default="linearregression", type=str)
    args = parser.parse_args()

    if args.method != "debiasclip":
        usingclip = True
        model, preprocess = clip.load("ViT-B/32", device=args.device)
    else:
        usingclip = False
        model, preprocess = dclip.load("ViT-B/16-gender", device =args.device) # DebiasClip for gender, the only publicly available model

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
        oracle = LinearRegression()
    else:
        print("Function class not supported.")
        exit()

    batch_size = 512

    dataset_path = "/n/holylabs/LABS/calmon_lab/Lab/datasets/"
    if usingclip and args.dataset+'_clipfeatures.npy' in os.listdir(dataset_path):
        print("clip features, labels already processed")
        features = np.load(dataset_path+args.dataset+'_clipfeatures.npy')
        labels = np.load(dataset_path+args.dataset+'_cliplabels.npy')
    elif not usingclip and args.dataset+'_dclipFeatures.npy' in os.listdir(dataset_path):
        print("dclip features, labels already processed")
        features = np.load(dataset_path+args.dataset+'_dclipFeatures.npy')
        labels = np.load(dataset_path+args.dataset+'_dclipLabels.npy')
    else:
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
        if usingclip:
            np.save(dataset_path+ args.dataset+'_clipfeatures.npy', features)
            np.save(dataset_path+ args.dataset+'_cliplabels.npy', labels)
        else:
            np.save(dataset_path+ args.dataset+'_dclipFeatures.npy', features)
            np.save(dataset_path+ args.dataset+'_dclipLabels.npy', labels)       

    m = labels.shape[0]

    q = args.query

    q_token = clip.tokenize(q).to(args.device)

    # ensure on the same device
    model = model.to(args.device)
    q_token = q_token.to(args.device)

    with torch.no_grad():
        q_emb = model.encode_text(q_token).cpu().numpy().astype(np.float64)
    q_emb = q_emb/np.linalg.norm(q_emb)

    features = features.astype(np.float64)
    features = features/ np.linalg.norm(features, axis=1, keepdims=True)  # Calculate L2 norm for each row

    # compute similarities
    s = features @ q_emb.T

    results = {}
    if args.method == "mmr":
        solver = MMR(s, labels, features)
        lambdas = np.linspace(0, 1-1e-5, 50)

        reps = []
        sims = []
        indices_list = []
        selection_list = []
        MMR_cost_list = []
        for p in tqdm(lambdas):
            indices, diversity_cost, selection = solver.fit(args.k, p) 
            rep = getMPR(indices, labels, oracle, args.k, m)
            sim = (s.T @ indices)
            reps.append(rep)
            sims.append(sim)
            indices_list.append(indices)
            selection_list.append(selection)
            MMR_cost_list.append(diversity_cost)

        results['MPR'] = reps
        results['sims'] = sims
        results['indices'] = indices_list
        results['lambdas'] = lambdas
        results['selection'] = selection_list
        results['MMR_cost'] = MMR_cost_list
    
    elif args.method == "debiasclip":
        # return top k similarities
        top_indices = np.zeros(m)
        selection = np.argsort(s.squeeze())[::-1][:args.k]
        top_indices[selection] = 1
        sims = s.T@top_indices

        reps = getMPR(top_indices, labels, oracle, args.k, m)
        AssertionError(np.sum(top_indices)==args.k)
        results['sims'] = sims
        results['selection'] = selection
        results['indices'] = top_indices
        results['MPR'] = reps
    
    elif args.method == "clipclip":
        # get the order of columns to drop to reduce MI with sensitive attributes
        sensitive_attributes_idx = [dataset.attr_to_idx['Male']]
        gender_MI_order = return_feature_MI_order(features, labels, sensitive_attributes_idx)
        # run clipclip method
        solver = ClipClip(features, gender_MI_order, args.device)

        cols_drop = list(range(1, 400, 10))
        reps = []
        sims = []
        indices_list = []
        selection_list = []
        # drop a range of columns
        for num_col in tqdm(cols_drop):
            indices, selection = solver.fit(args.k, num_col,q_emb) 
            rep = getMPR(indices, labels, oracle, args.k, m)
            sim = (s.T @ indices)
            reps.append(rep)
            sims.append(sim)
            indices_list.append(indices)
            selection_list.append(selection)

        results['MPR'] = reps
        results['sims'] = sims
        results['indices'] = indices_list
        results['lambdas'] = cols_drop # number of columns dropped
        results['selection'] = selection_list

    elif args.method == "pbm":
        pbm_classes = [0, 1, 2] # correspond to predicted sensitive attribute being [Neutral/Uncertain, Male, Female]. 
        classes = ["Neither male nor female", "Male", "Female"]
        class_embeddings = []
        for text in classes:
            class_token = clip.tokenize(text).to(args.device)
            with torch.no_grad():
                class_emb = model.encode_text(class_token).cpu().numpy().astype(np.float64)
            class_embeddings.append(class_emb/np.linalg.norm(class_emb))
        pbm_labels = features @ (np.array(class_embeddings).squeeze().T)
        # select the highest value as predicted labels
        pbm_labels = np.argmax(pbm_labels, axis=1)
        print(np.unique(pbm_labels, return_counts=True))

        solver = PBM(features, s, pbm_labels, pbm_classes)
        lambdas = np.linspace(1e-5, 1-1e-5, 50)
        reps = []
        sims = []
        indices_list = []
        selection_list = []
        # drop a range of columns
        for eps in tqdm(lambdas):
            indices, selection = solver.fit(args.k, eps) 
            rep = getMPR(indices, labels, oracle, args.k, m)
            sim = (s.T @ indices)
            reps.append(rep)
            sims.append(sim)
            indices_list.append(indices)
            selection_list.append(selection)

        results['MPR'] = reps
        results['sims'] = sims
        results['indices'] = indices_list
        results['lambdas'] = lambdas #control amt of intervention. eps = .5 means half the time you take a PBM step, half the time you take a greedy one
        results['selection'] = selection_list


    result_path = './results/carol/'
    filename_pkl = "{}_{}_{}_{}.pkl".format(args.dataset, args.method, args.k, args.functionclass)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path + filename_pkl, 'wb') as f:
        pickle.dump(results, f)
    
    plt.figure()
    plt.plot(reps, sims, label="Binary")
    plt.xlabel('Representation')
    plt.ylabel('Similarity')
    plt.savefig("./results/carol/mmr_rep_sim.png")
if __name__ == "__main__":
    main()