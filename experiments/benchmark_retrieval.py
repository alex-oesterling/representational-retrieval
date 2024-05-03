from representational_retrieval import *
import torch
import numpy as np
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
    parser.add_argument('-curation_dataset', default=None, type=str)
    parser.add_argument('-n_samples', default=10000, type=int)
    parser.add_argument('-n_curation_samples', default=0, type=int)
    parser.add_argument('-query', default="queries.txt", type=str)
    parser.add_argument('-k', default=10, type=int)
    parser.add_argument('-functionclass', default="linearregression", type=str)
    args = parser.parse_args()
    print(args)

    if args.method != "debiasclip":
        embedding_model = "clip"
        model, preprocess = clip.load("ViT-B/32", device=args.device)
    else:
        embedding_model = "debiasclip"
        model, preprocess = dclip.load("ViT-B/16-gender", device =args.device) # DebiasClip for gender, the only publicly available model

    # Load the dataset
    if args.dataset == "fairface":
        dataset = FairFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", train=True, transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "occupations":
        dataset = Occupations("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "utkface":
        dataset = UTKFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "celeba":
        dataset = CelebA("/n/holylabs/LABS/calmon_lab/Lab/datasets/", attributes=None, train=True, transform=preprocess, embedding_model=embedding_model)
    else:
        print("Dataset not supported!")
        exit()

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

    if args.functionclass == "randomforest":
        reg_model = RandomForestRegressor(max_depth=2)
    elif args.functionclass == "linearregression":
        reg_model = LinearRegression()
    else:
        print("Function class not supported.")
        exit()

    batch_size = 512
    if args.n_samples == -1:
        args.n_samples = len(dataset)
    elif args.n_samples > len(dataset):
        print("n_samples greater than dataset size. Please put -1 for full dataset. Breaking.")
        exit()

    ## This is slow becase appending tensors to list. If we want to speed this up, precompute retrieval_features size
    # save features and labels
    label_size = dataset[0][1].shape[-1]
    num_batches = (args.n_samples // batch_size) + int(args.n_samples % batch_size != 0)
    retrieval_features = torch.zeros((num_batches*batch_size, 512))
    retrieval_labels = torch.zeros((num_batches*batch_size, label_size))
    ix = 0
    for _, labels, embedding in tqdm(DataLoader(dataset, batch_size=batch_size)):
        retrieval_features[ix*batch_size:(ix+1)*batch_size, :] = embedding
        retrieval_labels[ix*batch_size:(ix+1)*batch_size, :] = labels
        ix += 1
        if ix>=num_batches:
            break

    retrieval_features, retrieval_labels =retrieval_features.cpu().numpy(), retrieval_labels.cpu().numpy()

    retrieval_features = retrieval_features.astype(np.float64)
    retrieval_features = retrieval_features/ np.linalg.norm(retrieval_features, axis=1, keepdims=True)  # Calculate L2 norm for each row

    ## Do the exact same for the curation set if it exists
    if curation_dataset is not None:
        if args.n_curation_samples == -1:
            args.n_curation_samples = len(curation_dataset)
        elif args.n_curation_samples > len(curation_dataset):
            print("n_samples greater than dataset size. Please put -1 for full dataset. Breaking.")
            exit()

        ## This is slow becase appending tensors to list. If we want to speed this up, precompute retrieval_features size
        # save features and labels
        label_size = dataset[0][1].shape[-1]
        num_batches = (args.n_curation_samples // batch_size) + int(args.n_curation_samples % batch_size != 0)
        curation_features = torch.zeros(num_batches*batch_size, 512)
        curation_labels = torch.zeros(num_batches*batch_size, label_size)
        ix = 0
        for _, labels, embedding in tqdm(DataLoader(curation_dataset, batch_size=batch_size)):
            curation_features[ix*batch_size:(ix+1)*batch_size, :] = embedding
            curation_labels[ix*batch_size:(ix+1)*batch_size, :] = labels
            ix += 1
            if ix >= num_batches:
                break

        curation_features, curation_labels = curation_features.cpu().numpy(), curation_labels.cpu().numpy()
        curation_features = curation_features.astype(np.float64)
        curation_features = curation_features/ np.linalg.norm(curation_features, axis=1, keepdims=True)  # Calculate L2 norm for each row
    else:
        curation_features = None
        curation_labels = None

    # dataset_path = "/n/holylabs/LABS/calmon_lab/Lab/datasets/"
    # if usingclip and args.dataset+'_normalized_clipfeatures_{}.npy'.format(args.n_samples) in os.listdir(dataset_path):
    #     print("normalized clip features, labels already processed")
    #     features = np.load(dataset_path+args.dataset+'_normalized_clipfeatures_{}.npy'.format(args.n_samples))
    #     labels = np.load(dataset_path+args.dataset+'_cliplabels_{}.npy'.format(args.n_samples))
    # elif not usingclip and args.dataset+'_normalized_dclipFeatures_{}.npy'.format(args.n_samples) in os.listdir(dataset_path):
    #     print("normalized dclip features, labels already processed")
    #     features = np.load(dataset_path+args.dataset+'_normalized_dclipFeatures_{}.npy'.format(args.n_samples))
    #     labels = np.load(dataset_path+args.dataset+'_dclipLabels_{}.npy'.format(args.n_samples))
    # else:
    #     all_features = []
    #     all_labels = []

    n = retrieval_labels.shape[0]

    with open(args.query, 'r') as f:
        queries = f.readlines()
    model = model.to(args.device)

    for query in tqdm(queries[::-1]):
        q_org = query.strip()
        q = "A photo of "+ q_org
        print(q)
        q_token = clip.tokenize(q).to(args.device)

        # ensure on the same device
        # q_token = q_token.to(args.device)

        with torch.no_grad():
            q_emb = model.encode_text(q_token).cpu().numpy().astype(np.float64)
        q_emb = q_emb/np.linalg.norm(q_emb)

        # compute similarities
        s = retrieval_features @ q_emb.T

        del q_emb
        torch.cuda.empty_cache()

        results = {}
        if args.method == "lp":
            solver = GurobiLP(s, retrieval_labels, curation_set=curation_labels, model=reg_model)

            num_iter = 10

            reps = []
            sims = []
            rounded_reps = []
            rounded_sims = []
            indices_list = []
            rounded_indices_list = []
            # rhoslist = []
            rhos = np.linspace(0.005, 0.025, 20)
            for rho in tqdm(rhos, desc="rhos"):
                indices = solver.fit(args.k, num_iter, rho)
                if indices is None: ## returns none if problem is infeasible
                    break
                indices_rounded = indices.copy()
                indices_rounded[np.argsort(indices_rounded)[::-1][args.k:]] = 0
                indices_rounded[indices_rounded>1e-5] = 1.0 

                rep = getMPR(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
                sim = (s.T @ indices)

                reps.append(rep)
                sims.append(sim[0])
                indices_list.append(indices)

                rounded_rep = getMPR(indices_rounded, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
                rounded_sim = (s.T @ indices_rounded)

                rounded_reps.append(rounded_rep)
                rounded_sims.append(rounded_sim[0])
                rounded_indices_list.append(indices_rounded)

                # rhoslist.append(rho)

            # for p in tqdm(lambdas):
            #     indices, diversity_cost, selection = solver.fit(args.k, num_iter) 
            #     # rep = getMPR(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
            #     sim = (s.T @ indices)
            #     reps.append(rep)
            #     sims.append(sim)
            #     indices_list.append(indices)
            #     selection_list.append(selection)
            #     MMR_cost_list.append(diversity_cost)

            results['MPR'] = reps
            results['sims'] = sims
            results['indices'] = indices_list
            results['rounded_MPR'] = rounded_reps
            results['rounded_sims'] = rounded_sims
            results['rounded_indices'] = rounded_indices_list
            results['rhos'] = rhos
            # results['selection'] = selection_list
            # results['MMR_cost'] = MMR_cost_list
        elif args.method == "ip":
            solver = GurobiIP(s, retrieval_labels, curation_set=curation_labels, model = reg_model)

            num_iter = 10

            reps = []
            sims = []
            rounded_reps = []
            rounded_sims = []
            indices_list = []
            rounded_indices_list = []
            # rhoslist = []
            rhos = np.linspace(0.005, 0.025, 20)
            for rho in tqdm(rhos, desc="rhos"):
                indices = solver.fit(args.k, num_iter, rho)
                
                if indices is None: ## returns none if problem is infeasible
                    break
                sparsity = sum(indices>1e-4)
                
                rep = getMPR(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
                sim = (s.T @ indices)

                reps.append(rep)
                sims.append(sim[0])
                indices_list.append(indices)

            results['MPR'] = reps
            results['sims'] = sims
            results['indices'] = indices_list
            results['rhos'] = rhos
        elif args.method == "mmr":
            solver = MMR(s, retrieval_features, curation_embeddings=curation_features)
            lambdas = np.linspace(0, 1-1e-5, 50)

            reps = []
            sims = []
            indices_list = []
            selection_list = []
            MMR_cost_list = []
            for p in tqdm(lambdas):
                indices, diversity_cost, selection = solver.fit(args.k, p) 
                rep = getMPR(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
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
        
        elif args.method == "mmr_mpr":
            solver = MMR_MPR(s, retrieval_labels, curation_set=curation_labels, model=reg_model)
            lambdas = np.linspace(0, 1-1e-5, 20)
            
            reps = []
            sims = []
            indices_list = []
            selection_list = []
            for p in tqdm(lambdas):
                indices, selection = solver.fit(args.k, p) 
                rep = getMPR(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
                sim = (s.T @ indices)
                reps.append(rep)
                sims.append(sim)
                indices_list.append(indices)
                selection_list.append(selection)

                results['MPR'] = reps
                results['sims'] = sims
                results['indices'] = indices_list
                results['lambdas'] = lambdas
                results['selection'] = selection_list


        elif args.method == "debiasclip":
            # return top k similarities
            top_indices = np.zeros(n)
            selection = np.argsort(s.squeeze())[::-1][:args.k]
            top_indices[selection] = 1
            sims = s.T@top_indices

            reps = getMPR(top_indices, retrieval_labels, args.k, curation_set = curation_labels, model=reg_model)
            AssertionError(np.sum(top_indices)==args.k)
            results['sims'] = sims
            results['selection'] = selection
            results['indices'] = top_indices
            results['MPR'] = reps
        
        elif args.method == "clipclip":
            # get the order of columns to drop to reduce MI with sensitive attributes (support intersectional groups)
            sensitive_attributes_idx = [dataset.attr_to_idx['Male']]
            if curation_dataset is not None:
                gender_MI_oder = return_feature_MI_order(curation_features, curation_labels, sensitive_attributes_idx)
            else:
                gender_MI_order = return_feature_MI_order(retrieval_features, retrieval_labels, sensitive_attributes_idx)
            # run clipclip method
            solver = ClipClip(retrieval_features, gender_MI_order, args.device)

            cols_drop = list(range(1, 400, 10))
            reps = []
            sims = []
            indices_list = []
            selection_list = []
            # drop a range of columns
            for num_col in tqdm(cols_drop):
                indices, selection = solver.fit(args.k, num_col,q_emb) 
                rep = getMPR(indices, retrieval_labels, args.k, curation_set = curation_labels, model=reg_model)
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
            pbm_labels = retrieval_features @ (np.array(class_embeddings).squeeze().T)
            # select the highest value as predicted labels
            pbm_labels = np.argmax(pbm_labels, axis=1)
            print(np.unique(pbm_labels, return_counts=True))

            solver = PBM(retrieval_features, s, pbm_labels, pbm_classes)
            lambdas = np.linspace(1e-5, 1-1e-5, 50)
            reps = []
            sims = []
            indices_list = []
            selection_list = []
            # drop a range of columns
            for eps in tqdm(lambdas):
                indices, selection = solver.fit(args.k, eps) 
                rep = getMPR(indices, retrieval_labels, args.k, curation_set = curation_labels, model=reg_model)
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
            
        del solver

        result_path = './results/alex/'
        q_title = q.split(" ")[-1]
        filename_pkl = "{}_{}_{}_{}_{}.pkl".format(args.dataset, args.method, args.k, args.functionclass, q_title)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(result_path + filename_pkl, 'wb') as f:
            pickle.dump(results, f)

    print(args)

if __name__ == "__main__":
    main()