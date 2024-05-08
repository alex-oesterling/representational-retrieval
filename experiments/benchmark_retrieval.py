import sys
sys.path.append("../")
from representational_retrieval import *
import torch
import numpy as np
from tqdm.auto import tqdm
import argparse
import seaborn as sns
import clip
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
sns.set_style("whitegrid")
import pickle
import os
import debias_clip as dclip
import json

def get_top_embeddings_labels_ids(dataset, query, embedding_model, datadir):
    if datadir == "occupations": ## Occupations is so small no need to use all 10k
        embeddings = []
        filepath = "/n/holylabs/LABS/calmon_lab/Lab/datasets/occupations/"
        for i, path in enumerate(dataset.img_paths):
            image_id = os.path.relpath(path.split(".")[0], filepath)
            embeddingpath = os.path.join(filepath, embedding_model, image_id+".pt")
            embeddings.append(torch.load(embeddingpath))
        embeddings = torch.stack(embeddings).cpu().numpy()
        labels = dataset.labels.numpy()
        indices = torch.arange(embeddings.shape[0])
    else:
        retrievaldir = os.path.join("/n/holylabs/LABS/calmon_lab/Lab/datasets", datadir, embedding_model,query)
        embeddings = np.load(os.path.join(retrievaldir, "embeds.npy"))
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
    parser.add_argument('-method', default="mmr", type=str)
    parser.add_argument('-device', default="cuda", type=str)
    parser.add_argument('-dataset', default="celeba", type=str)
    parser.add_argument('-curation_dataset', default=None, type=str)
    parser.add_argument('-query', default="queries.txt", type=str)
    parser.add_argument('-k', default=10, type=int)
    parser.add_argument('-functionclass', default="randomforest", type=str)
    parser.add_argument('-use_clip', action="store_true")
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
    elif args.functionclass == "decisiontree":
        reg_model = DecisionTreeRegressor(max_depth=5)
    elif args.functionclass == "mlp":
        reg_model = MLPRegressor([64, 64])
    elif args.functionclass == "linearregressiontheoretical":
        reg_model = "linearregressiontheoretical"
    else:
        print("Function class not supported.")
        exit()

    # batch_size = 512
    # if args.n_samples == -1:
    #     args.n_samples = len(dataset)
    # elif args.n_samples > len(dataset):
    #     print("n_samples greater than dataset size. Please put -1 for full dataset. Breaking.")
    #     exit()

    ## This is slow becase appending tensors to list. If we want to speed this up, precompute retrieval_features size
    # save features and labels
    # label_size = dataset[0][1].shape[-1]
    # num_batches = (args.n_samples // batch_size) + int(args.n_samples % batch_size != 0)
    # retrieval_features = torch.zeros((num_batches*batch_size, 512))
    # retrieval_labels = torch.zeros((num_batches*batch_size, label_size))
    # ix = 0
    # for _, labels, embedding in tqdm(DataLoader(dataset, batch_size=batch_size)):
    #     retrieval_features[ix*batch_size:(ix+1)*batch_size, :] = embedding
    #     retrieval_labels[ix*batch_size:(ix+1)*batch_size, :] = labels
    #     ix += 1
    #     if ix>=num_batches:
    #         break

    # retrieval_features, retrieval_labels = retrieval_features.cpu().numpy(), retrieval_labels.cpu().numpy()

    # retrieval_features = retrieval_features.astype(np.float64)
    # retrieval_features = retrieval_features/ np.linalg.norm(retrieval_features, axis=1, keepdims=True)  # Calculate L2 norm for each row

    ## Do the exact same for the curation set if it exists
    # if curation_dataset is not None:
        # if args.n_curation_samples == -1:
        #     args.n_curation_samples = len(curation_dataset)
        # elif args.n_curation_samples > len(curation_dataset):
        #     print("n_samples greater than dataset size. Please put -1 for full dataset. Breaking.")
        #     exit()

        # ## This is slow becase appending tensors to list. If we want to speed this up, precompute retrieval_features size
        # # save features and labels
        # label_size = curation_dataset[0][1].shape[-1]
        # num_batches = (args.n_curation_samples // batch_size) + int(args.n_curation_samples % batch_size != 0)
        # curation_features = torch.zeros(num_batches*batch_size, 512)
        # curation_labels = torch.zeros(num_batches*batch_size, label_size)
        # ix = 0
        # for _, labels, embedding in tqdm(DataLoader(curation_dataset, batch_size=batch_size)):
        #     curation_features[ix*batch_size:(ix+1)*batch_size, :] = embedding
        #     curation_labels[ix*batch_size:(ix+1)*batch_size, :] = labels
        #     ix += 1
        #     if ix >= num_batches:
        #         break

        # curation_features, curation_labels = curation_features.cpu().numpy(), curation_labels.cpu().numpy()
        # curation_features = curation_features.astype(np.float64)
        # curation_features = curation_features/ np.linalg.norm(curation_features, axis=1, keepdims=True)  # Calculate L2 norm for each row
        # ## Prune labels to ones shared by both datasets:
        # with open('representational_retrieval/shared_labels.json') as f:
        #     d = json.load(f)[args.dataset][args.curation_dataset]
        #     print("Shared labels for {}, {}:".format(args.dataset, args.curation_dataset))
        #     ret_indices = []
        #     cur_indices = []
        #     for lab in d.keys():
        #         print(lab)
        #         ret_indices.extend(d[lab][0])
        #         cur_indices.extend(d[lab][1])
        #     print(ret_indices)
        #     print(cur_indices)
        # curation_labels = curation_labels[:, cur_indices]
        # retrieval_labels = retrieval_labels[:, ret_indices]
                
                
    # else:
    #     curation_features = None
    #     curation_labels = None

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


    with open(args.query, 'r') as f:
        queries = f.readlines()
    model = model.to(args.device)

    for query in tqdm(queries):
        q_org = query.strip()
        q = "A photo of "+ q_org
        q_tag = q.split(" ")[-1]
        print(q)
        q_token = clip.tokenize(q).to(args.device)

        retrieval_features, retrieval_labels, retrieval_indices = get_top_embeddings_labels_ids(
            dataset,
            q_tag,
            embedding_model,
            args.dataset
        )
        
        if curation_dataset is not None:
            curation_features, curation_labels, curation_indices = get_top_embeddings_labels_ids(
                curation_dataset,
                q_tag,
                embedding_model,
                args.curation_dataset
            )
            with open('representational_retrieval/shared_labels.json') as f:
                d = json.load(f)[args.dataset][args.curation_dataset]
                print("Shared labels for {}, {}:".format(args.dataset, args.curation_dataset))
                ret_indices = []
                cur_indices = []
                for lab in d.keys():
                    print(lab)
                    ret_indices.append(d[lab][0]) if isinstance(d[lab][0], int) else ret_indices.extend(d[lab][0])
                    cur_indices.append(d[lab][1]) if isinstance(d[lab][1], int) else cur_indices.extend(d[lab][1])
            print(ret_indices)
            print(cur_indices)
            curation_labels_full = curation_labels
            curation_labels = curation_labels[:, cur_indices]
            retrieval_labels = retrieval_labels[:, ret_indices]
        else:
            curation_features = None
            curation_labels = None
            curation_labels_full = None

        n = retrieval_labels.shape[0]

        if args.use_clip:
            curation_labels = curation_features
            retrieval_labels = retrieval_features 

        print(retrieval_features.shape)

        with torch.no_grad():
            q_emb = model.encode_text(q_token).cpu().numpy().astype(np.float64)
        q_emb = q_emb/np.linalg.norm(q_emb)

        # compute similarities
        s = retrieval_features @ q_emb.T

        torch.cuda.empty_cache()

        results = {}
        if args.method == "lp":
            solver = GurobiLP(s, retrieval_labels, curation_set=curation_labels, model=reg_model)

            num_iter = 50

            reps = []
            sims = []
            rounded_reps = []
            rounded_sims = []
            indices_list = []
            rounded_indices_list = []
            rhos = np.linspace(0.01, 0.02, 40)
            for rho in tqdm(rhos, desc="rhos"):
                indices = solver.fit(args.k, num_iter, rho)
                if indices is None: ## returns none if problem is infeasible
                    continue
                indices_rounded = indices.copy()
                indices_rounded[np.argsort(indices_rounded)[::-1][args.k:]] = 0
                indices_rounded[indices_rounded>1e-5] = 1.0 

                rep = getMPR(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
                print("Rep: ", rep)
                sim = (s.T @ indices)

                reps.append(rep)
                sims.append(sim[0])
                indices_list.append(indices)

                rounded_rep = getMPR(indices_rounded, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
                print("Rounded Rep", rounded_rep)
                rounded_sim = (s.T @ indices_rounded)

                rounded_reps.append(rounded_rep)
                rounded_sims.append(rounded_sim[0])
                rounded_indices_list.append(indices_rounded)

            results['MPR'] = reps
            results['sims'] = sims
            results['indices'] = indices_list
            results['rounded_MPR'] = rounded_reps
            results['rounded_sims'] = rounded_sims
            results['rounded_indices'] = rounded_indices_list
            results['rhos'] = rhos
            solver.problem.dispose()
            del solver
        elif args.method == "ip":
            solver = GurobiIP(s, retrieval_labels, curation_set=curation_labels, model = reg_model)

            num_iter = 50

            reps = []
            sims = []
            rounded_reps = []
            rounded_sims = []
            indices_list = []
            rounded_indices_list = []
            rhos = np.linspace(0.005, 0.025, 20)
            for rho in tqdm(rhos, desc="rhos"):
                indices = solver.fit(args.k, num_iter, rho)
                
                if indices is None: ## returns none if problem is infeasible
                    continue
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
            solver.problem.dispose()
            del solver

        elif args.method == "mmr":
            solver = MMR(s, retrieval_features, curation_embeddings=curation_features)
            lambdas = np.linspace(0, 1-1e-5, 50)

            reps = []
            sims = []
            indices_list = []
            selection_list = []
            MMR_cost_list = []
            old_index = None
            for p in tqdm(lambdas):
                indices, diversity_cost, selection = solver.fit(args.k, p) 
                # print(indices, flush=True)
                rep = getMPR(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
                sim = (s.T @ indices)
                reps.append(rep)
                sims.append(sim)
                print(sim, flush=True)
                indices_list.append(indices)
                selection_list.append(selection)
                MMR_cost_list.append(diversity_cost)
                if old_index is None:
                    old_index = indices
                else:
                    print(np.sum(indices - old_index), flush=True)
                    old_index = indices

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
            # print(s)
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
            if curation_dataset is not None:
                if args.dataset == "occupations":
                    gender_idx = 0
                elif args.dataset == "celeba":
                    gender_idx = 20
                elif args.dataset == "fairface":
                    gender_idx = 0
                elif args.dataset == "utkface":
                    gender_idx = 0
                gender_MI_order = return_feature_MI_order(curation_features, curation_labels_full, [gender_idx])
            else:
                if args.dataset == "occupations":
                    gender_idx = 0
                elif args.dataset == "celeba":
                    gender_idx = 20
                elif args.dataset == "fairface":
                    gender_idx = 0
                elif args.dataset == "utkface":
                    gender_idx = 0
                gender_MI_order = return_feature_MI_order(retrieval_features, retrieval_labels, [gender_idx]) ## This is for CelebA
                
            # run clipclip method
            solver = ClipClip(retrieval_features, gender_MI_order, args.device)

            cols_drop = list(range(1, 400, 10))
            reps = []
            sims = []
            indices_list = []
            selection_list = []
            # drop a range of columns
            for num_col in tqdm(cols_drop):
                print(num_col)
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
        result_path = './results/alex/'
        q_title = q.split(" ")[-1]
        if args.use_clip:
            filename_pkl = "clip_{}_curation_{}_top10k_{}_{}_{}_{}.pkl".format(args.dataset, args.curation_dataset, args.method, args.k, args.functionclass, q_title)
        else:
            filename_pkl = "{}_curation_{}_top10k_{}_{}_{}_{}.pkl".format(args.dataset, args.curation_dataset, args.method, args.k, args.functionclass, q_title)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(result_path + filename_pkl, 'wb') as f:
            pickle.dump(results, f)

    print(args)

if __name__ == "__main__":
    main()