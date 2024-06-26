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

    # np.random.seed(1017)
    # print(sklearn.utils.check_random_state())

    if args.method != "debiasclip":
        embedding_model = "clip"
        # model, preprocess = clip.load("ViT-B/32", device=args.device)
        model, preprocess = clip.load("ViT-B/32", device=args.device)
        # model = model.to(args.device)

    else:
        embedding_model = "debiasclip"
        model, preprocess = dclip.load("ViT-B/16-gender", device =args.device) # DebiasClip for gender, the only publicly available model
        # model = model.to(args.device)

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
        reg_model = RandomForestRegressor(max_depth=2, random_state=1017)
    elif args.functionclass == "linearregression":
        reg_model = LinearRegression(fit_intercept=False)
    elif args.functionclass == "decisiontree":
        reg_model = DecisionTreeRegressor(max_depth=3, random_state=1017)
    elif args.functionclass == "mlp":
        reg_model = MLPRegressor([64], random_state=1017)
    else:
        print("Function class not supported.")
        exit()

    with open(args.query, 'r') as f:
        queries = f.readlines()

    for query in tqdm(queries):
        q_org = query.strip()
        q = "A photo of "+ q_org
        q_tag = " ".join(q.split(" ")[4:])
        print(q_tag)
        q_emb = np.load("representational_retrieval/queries/{}_{}.npy".format(embedding_model, q_tag))
        # q_token = clip.tokenize(q).to(args.device)

        retrieval_features, retrieval_labels, retrieval_indices, retrieval_probe_labels = get_top_embeddings_labels_ids(
            dataset,
            q_tag,
            embedding_model,
            args.dataset
        )
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

            # with open('representational_retrieval/shared_labels.json') as f:
            #     d = json.load(f)[args.dataset][args.curation_dataset]
            #     print("Shared labels for {}, {}:".format(args.dataset, args.curation_dataset))
            #     ret_indices = []
            #     cur_indices = []
            #     for lab in d.keys():
            #         print(lab)
            #         ret_indices.append(d[lab][0]) if isinstance(d[lab][0], int) else ret_indices.extend(d[lab][0])
            #         cur_indices.append(d[lab][1]) if isinstance(d[lab][1], int) else cur_indices.extend(d[lab][1])
            # print(ret_indices)
            # print(cur_indices)
            # curation_labels = curation_labels[:, cur_indices]
            # retrieval_labels = retrieval_labels[:, ret_indices]
            curation_labels_full = curation_labels

            if args.dataset == "utkface": ## remap races to utkface
                new_races = np.zeros((curation_probe_labels.shape[0], 5))
                new_races[:, 0] = curation_probe_labels[:, 5]
                new_races[:, 1] = curation_probe_labels[:, 4]
                new_races[:, 2] = np.logical_or(curation_probe_labels[:, 2], curation_probe_labels[:, 8]) ## Asian, SE Asian
                new_races[:, 3] = curation_probe_labels[:, 3]
                new_races[:, 4] = np.logical_or(curation_probe_labels[:, 6], curation_probe_labels[:, 7]) ## Middle Eastern, Latino
                curation_probe_labels = np.concatenate((curation_probe_labels[:, :2], new_races), axis=1)
            if args.curation_dataset == "utkface":
                new_races = np.zeros((retrieval_probe_labels.shape[0], 5))
                new_races[:, 0] = retrieval_probe_labels[:, 5]
                new_races[:, 1] = retrieval_probe_labels[:, 4]
                new_races[:, 2] = np.logical_or(retrieval_probe_labels[:, 2], retrieval_probe_labels[:, 8]) ## Asian, SE Asian
                new_races[:, 3] = retrieval_probe_labels[:, 3]
                new_races[:, 4] = np.logical_or(retrieval_probe_labels[:, 6], retrieval_probe_labels[:, 7]) ## Middle Eastern, Latino
                retrieval_probe_labels = np.concatenate((retrieval_probe_labels[:, :2], new_races), axis=1)

            # synthetic_curation_probe_labels = None
            # num_races = curation_probe_labels.shape[1]-2
            # for i in range(2):
            #     for j in range(2):
            #         temp = np.concatenate([np.ones((num_races, 1))*i,np.ones((num_races, 1))*j,np.eye(num_races)], axis=1)
            #         if synthetic_curation_probe_labels is None:
            #             synthetic_curation_probe_labels = temp
            #         else:
            #             synthetic_curation_probe_labels = np.concatenate([synthetic_curation_probe_labels, temp], axis=0)

            # curation_labels = synthetic_curation_probe_labels
            curation_labels = curation_probe_labels
            retrieval_labels = retrieval_probe_labels
        else:
            curation_features = None
            curation_labels = None
            curation_labels_full = None

        n = retrieval_labels.shape[0]

        if args.use_clip:
            curation_labels = curation_features
            retrieval_labels = retrieval_features 

        print(retrieval_features.shape)

        # with torch.no_grad():
        #     q_emb = model.encode_text(q_token).cpu().numpy().astype(np.float64)
        # q_emb = q_emb/np.linalg.norm(q_emb)

        # compute similarities
        s = retrieval_features @ q_emb.T

        top_indices = np.zeros(n)
        selection = np.argsort(s.squeeze())[::-1][:args.k]
        top_indices[selection] = 1
        sim_upper_bound = s.T@top_indices
        # rep_upper_bounds = []
        # for _ in range(10):
        #     rep_upper_bounds.append(getMPR(top_indices, retrieval_labels, args.k, curation_set = curation_labels, model=reg_model)[0])
        # print(rep_upper_bounds)
        # rep_upper_bound = np.mean(rep_upper_bounds)
        # print("sim_upper_bound, rep_upper_bound: {}, {}".format(sim_upper_bound, rep_upper_bound))
        rep_upper_bound, _ = getMPR(top_indices, retrieval_labels, args.k, curation_set = curation_labels, model=reg_model)
        rep_upper_bound_1, _ = getMPR(top_indices, retrieval_labels, args.k, curation_set = curation_labels, model=reg_model)
        print(rep_upper_bound == rep_upper_bound_1, "MPR consistent across two calls") 
        print("KNN selection", selection)
        print("mpr for KNN", rep_upper_bound_1)
        print("sim for KNN", sim_upper_bound)

        # random_indices = np.zeros(n)
        
        # random_sims = []
        # random_reps = []
        # for _ in range(10):
        #     random_selection = np.random.choice(np.arange(n), args.k, replace=False)
        #     random_indices[random_selection] = 1
        #     random_sim2 = s.T@random_indices
        #     random_reps_oracles = []
        #     for _ in range(10):
        #         random_MPR, _ = getMPR(random_indices, retrieval_labels, args.k, curation_set = curation_labels, model=reg_model)
        #         random_reps_oracles.append(random_MPR)
        #     random_reps.append(np.mean(random_reps_oracles))
        #     random_sims.append(random_sim2)
        # #print(random_reps)
        # random_rep = np.mean(random_reps)
        # random_sim = np.mean(random_sims)
        #print("sim_random, rep_random: {}, {}".format(random_sim, random_rep))


        torch.cuda.empty_cache()

        results = {}

        if args.method == "lp":
            solver = GurobiLP(s, retrieval_labels, curation_set=curation_labels, model=reg_model)
            # if args.functionclass == "linearregression":
            #     closed_form_solver = BoundedDataNormLP(s, retrieval_labels, curation_labels=curation_labels)
            #     gt_reps = []
            #     gt_rounded_reps = []
            num_iter = 50

            reps = []
            sims = []
            # rounded_reps = []
            # rounded_sims = []
            indices_list = []
            # rounded_indices_list = []
            # lb, ub = get_lower_upper_bounds(args.k, s, retrieval_labels, curation_labels)
            #rhos = [rep_upper_bound]
            rhos = np.linspace(0.005, rep_upper_bound+1e-5, 50)
            #rhos = np.linspace(random_rep, rep_upper_bound, 50)
            # rhos = np.linspace(0.005, 0.025, 20)

            # rhos = np.linspace(lb, ub, 40)[::-1]
            for rho in tqdm(rhos, desc="rhos"):
                indices = solver.fit(args.k, num_iter, rho, top_indices)
                if indices is None: ## returns none if problem is infeasible
                    continue
                
                indices_rounded = indices.copy()
                indices_rounded[np.argsort(indices_rounded)[::-1][args.k:]] = 0
                indices_rounded[indices_rounded>1e-5] = 1.0 
                # print(np.where(indices_rounded==1)[0])
                # print(np.where(top_indices==1)[0])
                # assert (indices_rounded == top_indices).all(), f"Not starting from KNN indices"

                rep, _ = getMPR(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
                print("Rep: ", rep)
                sim = (s.T @ indices)
                print("Sim: ", sim)
                # if args.functionclass == "linearregression":
                #     gt_rep = closed_form_solver.getClosedMPR(indices, args.k)
                #     gt_reps.append(gt_rep)

                reps.append(rep)
                sims.append(sim[0])
                indices_list.append(indices)
                # if args.functionclass == "linearregression":
                #     gt_rounded_rep = closed_form_solver.getClosedMPR(indices_rounded, args.k)
                #     gt_rounded_reps.append(gt_rounded_rep)

                # rounded_rep = getMPR(indices_rounded, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
                # print("Rounded Rep", rounded_rep)
                # rounded_sim = (s.T @ indices_rounded)

                # rounded_reps.append(rounded_rep)
                # rounded_sims.append(rounded_sim[0])
                # rounded_indices_list.append(indices_rounded)

            results['MPR'] = reps
            results['sims'] = sims
            results['indices'] = indices_list
            # results['rounded_MPR'] = rounded_reps
            # results['rounded_sims'] = rounded_sims
            # results['rounded_indices'] = rounded_indices_list
            # results['rhos'] = rhos
            # if args.functionclass == "linearregression":
            #     results['gt_MPR'] = gt_reps
            #     results['gt_rounded_MPR'] = gt_rounded_reps
            if solver.problem:
                solver.problem.dispose()
            del solver
            # solver = GurobiLP(s, retrieval_labels, curation_set=curation_labels, model=reg_model)
            # # if args.functionclass == "linearregression":
            # #     closed_form_solver = BoundedDataNormLP(s, retrieval_labels, curation_labels=curation_labels)
            # #     gt_reps = []
            # #     gt_rounded_reps = []
            # num_iter = 50

            # reps = []
            # sims = []
            # reps.append(rep_upper_bound)
            # sims.append(sim_upper_bound.item())
            # rounded_reps = []
            # rounded_sims = []
            # indices_list = []
            # rounded_indices_list = []
            # # lb, ub = get_lower_upper_bounds(args.k, s, retrieval_labels, curation_labels)
            # #rhos = [rep_upper_bound]
            # rhos = np.linspace(0.005, rep_upper_bound+1e-5, 50)
            # #rhos = np.linspace(random_rep, rep_upper_bound, 50)
            # # rhos = np.linspace(0.005, 0.025, 20)

            # # rhos = np.linspace(lb, ub, 40)[::-1]
            # for rho in tqdm(rhos, desc="rhos"):
            #     indices = solver.fit(args.k, num_iter, rho, KNN_indices = top_indices)
            #     if indices is None: ## returns none if problem is infeasible
            #         continue
                
            #     indices_rounded = indices.copy()
            #     indices_rounded[np.argsort(indices_rounded)[::-1][args.k:]] = 0
            #     indices_rounded[indices_rounded>1e-5] = 1.0 
            #     # print(np.where(indices_rounded==1)[0])
            #     # print(np.where(top_indices==1)[0])
            #     # assert (indices_rounded == top_indices).all(), f"Not starting from KNN indices"

            #     rep, _ = getMPR(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
            #     print("Rep: ", rep)
            #     sim = (s.T @ indices)
            #     print("Sim: ", sim)
            #     # if args.functionclass == "linearregression":
            #     #     gt_rep = closed_form_solver.getClosedMPR(indices, args.k)
            #     #     gt_reps.append(gt_rep)

            #     reps.append(rep)
            #     sims.append(sim[0])
            #     indices_list.append(indices)

            #     rounded_rep, _ = getMPR(indices_rounded, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
            #     print("Rounded Rep", rounded_rep)
            #     # if args.functionclass == "linearregression":
            #     #     gt_rounded_rep = closed_form_solver.getClosedMPR(indices_rounded, args.k)
            #     #     gt_rounded_reps.append(gt_rounded_rep)

            #     rounded_sim = (s.T @ indices_rounded)

            #     rounded_reps.append(rounded_rep)
            #     rounded_sims.append(rounded_sim[0])
            #     rounded_indices_list.append(indices_rounded)
            #     if solver.problem:
            #         solver.problem.dispose()

            # results['MPR'] = reps
            # results['sims'] = sims
            # results['indices'] = indices_list
            # results['rounded_MPR'] = rounded_reps
            # results['rounded_sims'] = rounded_sims
            # results['rounded_indices'] = rounded_indices_list
            # results['rhos'] = rhos
            # # if args.functionclass == "linearregression":
            # #     results['gt_MPR'] = gt_reps
            # #     results['gt_rounded_MPR'] = gt_rounded_reps
            # if solver.problem:
            #     solver.problem.dispose()
            # # if closed_form_solver.problem:
            # #     closed_form_solver.problem.dispose()
            # del solver
            # # del closed_form_solver
        elif args.method == "ip":
            solver = GurobiIP(s, retrieval_labels, curation_set=curation_labels, model = reg_model)

            num_iter = 50

            reps = []
            sims = []
            rounded_reps = []
            rounded_sims = []
            indices_list = []
            rounded_indices_list = []
            rhos = np.linspace(random_rep, rep_upper_bound, 50)
            for rho in tqdm(rhos, desc="rhos"):
                indices = solver.fit(args.k, num_iter, rho)
                
                if indices is None: ## returns none if problem is infeasible
                    continue
                sparsity = sum(indices>1e-4)
                
                rep, _ = getMPR(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
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
        elif args.method == "closed_lp":
            if args.functionclass != "linearregression":
                print("linearregression required for closed lp")
                exit()
            # cuttingplanesolver = GurobiLP(s, retrieval_labels, curation_set=curation_labels, model=reg_model)
            solver = BoundedDataNormLP(s, retrieval_labels, curation_labels=curation_labels)

            lb, ub = solver.get_lower_upper_bounds(args.k)

            num_iter = 50

            reps = []
            cuttingplanereps = []
            sims = []
            rounded_reps = []
            rounded_cuttingplanereps = []
            rounded_sims = []
            indices_list = []
            rounded_indices_list = []
            rhos = np.linspace(lb, ub, 50)
            for rho in tqdm(rhos, desc="rhos"):
                indices = solver.fit(args.k, rho)
                indices_rounded = indices.copy()
                indices_rounded[np.argsort(indices_rounded)[::-1][args.k:]] = 0
                indices_rounded[indices_rounded>1e-5] = 1.0 

                rep = solver.getClosedMPR(indices, args.k)
                cuttingplanerep, c = getMPR(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
                print("norm Xw:", np.linalg.norm(c, axis=0))
                print("Rep: ", rep)
                sim = (s.T @ indices)
                print("Sim: ", sim)

                cuttingplanereps.append(cuttingplanerep)
                reps.append(rep)
                sims.append(sim[0])
                indices_list.append(indices)

                rounded_rep = solver.getClosedMPR(indices_rounded, args.k)
                rounded_cuttingplanerep, rounded_c = getMPR(indices_rounded, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)

                print("Rounded Rep", rounded_rep)
                print("norm Xw:", np.linalg.norm(rounded_c, axis=0))

                rounded_sim = (s.T @ indices_rounded)

                rounded_cuttingplanereps.append(rounded_cuttingplanerep)
                rounded_reps.append(rounded_rep)
                rounded_sims.append(rounded_sim[0])
                rounded_indices_list.append(indices_rounded)

            results['MPR'] = reps
            results['sims'] = sims
            results['indices'] = indices_list
            results['rounded_MPR'] = rounded_reps
            results['rounded_sims'] = rounded_sims
            results['rounded_indices'] = rounded_indices_list
            results['cutting_MPR'] = cuttingplanereps
            results['rounded_cutting_MPR'] = rounded_cuttingplanereps
            results['rhos'] = rhos
            del solver
        elif args.method == "mmr":
            solver = MMR(s, retrieval_features)
            # solver = MMR(s, retrieval_labels)
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
                rep, _ = getMPR(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
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
                rep, _ = getMPR(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
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

            reps, _ = getMPR(top_indices, retrieval_labels, args.k, curation_set = curation_labels, model=reg_model)
            AssertionError(np.sum(top_indices)==args.k)
            results['sims'] = sims
            results['selection'] = selection
            results['indices'] = top_indices
            results['MPR'] = reps
        
        elif args.method == "clipclip":
            # get the order of columns to drop to reduce MI with sensitive attributes (support intersectional groups)
            if curation_dataset is not None:
                gender_MI_order = return_feature_MI_order(curation_features, curation_labels_full, [0])
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
                rep, _ = getMPR(indices, retrieval_labels, args.k, curation_set = curation_labels, model=reg_model)
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
                rep, _ = getMPR(indices, retrieval_labels, args.k, curation_set = curation_labels, model=reg_model)
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
        result_path = '/n/holyscratch01/calmon_lab/Users/aoesterling/representational_retrieval/final_results/'
        q_title = q.split(" ")[-1]
        print("MPR: ", results['MPR'])
        print("sims: ", results['sims'])
        if args.use_clip:
            filename_pkl = "clip_{}_curation_{}_top10k_{}_{}_{}_{}.pkl".format(args.dataset, args.curation_dataset, args.method, args.k, args.functionclass, q_title)
        else:
            filename_pkl = "{}_curation_{}_top10k_{}_{}_{}_{}.pkl".format(args.dataset, args.curation_dataset, args.method, args.k, args.functionclass, q_title)
        print(filename_pkl)
        print(result_path+filename_pkl)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path,filename_pkl), 'wb') as f:
            pickle.dump(results, f)

    print(args)

if __name__ == "__main__":
    main()