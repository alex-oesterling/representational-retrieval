import numpy as np 
from sklearn.linear_model import LinearRegression
import sklearn.feature_selection as fs
import cvxpy as cp

def oracle_function(indices, dataset, curation_set=None, model=None):
    if model is None:
        model = LinearRegression()

    k = int(np.sum(indices))
    if curation_set is not None:
        m = curation_set.shape[0]
        expanded_dataset = np.concatenate((dataset, curation_set), axis=0)
        curation_indicator = np.concatenate((np.zeros(dataset.shape[0]), np.ones(curation_set.shape[0])))
        a_expanded = np.concatenate((indices, np.zeros(curation_set.shape[0])))
        m = curation_set.shape[0]
        alpha = (a_expanded/k - curation_indicator/m)
        reg = model.fit(expanded_dataset, alpha)
    else:
        m = dataset.shape[0]
        alpha = (indices/k - 1/m)
        reg = model.fit(dataset, alpha)
    return reg

def getMPR(indices, dataset, k, curation_set=None, model=None):
    # if model is not None:
    #     if model == "linearrkhs":
    #         if curation_set is not None:
    #             m = curation_set.shape[0]
    #             print((dataset*indices).shape, flush=True)
    #             term1 = 1/(k**2) * np.sum((dataset*indices)@(dataset*indices).T)
    #             term2 = 2/(k*m) * np.sum((dataset*indices)@curation_set.T)
    #             term3 = 1/(m**2) * np.sum(curation_set@curation_set.T)
    #             mpr = np.sqrt(term1-term2+term3)
    #         else:
    #             m = dataset.shape[0]
    #             term1 = 1/(k**2) * np.sum((dataset*indices)@(dataset*indices).T)
    #             term2 = 2/(k*m) * np.sum((dataset*indices)@dataset.T)
    #             term3 = 1/(m**2) * np.sum(dataset@dataset.T)
    #             mpr = np.sqrt(term1-term2+term3)
    #         return mpr

    reg = oracle_function(indices, dataset, curation_set=curation_set, model=model)

    if curation_set is not None:
        curation_set = dataset
    expanded_dataset = np.concatenate((dataset, curation_set), axis=0)
    m = curation_set.shape[0]
    c = reg.predict(expanded_dataset)
    c /= np.linalg.norm(c)
    c *= c.shape[0]
    mpr = np.abs(np.sum((indices/k)*c[:dataset.shape[0]]) - np.sum((1/m)*c[dataset.shape[0]:]))
    # else:
    #     m = dataset.shape[0]
    #     c = reg.predict(dataset)
    #     c /= np.linalg.norm(c)
    #     c *= c.shape[0]
    #     mpr = np.abs(np.sum((indices/k)*c) - np.sum((1/m)*c))
    
    return mpr, c

def MMR_cost(indices, embeddings):
    # compute the diversity score of the entire set
    marginal_diversity = 0
    subset = embeddings[indices==1]
    mean_embedding, std_embedding = statEmbedding(embeddings)
    for i in range(len(subset)):
        for j in range(i+1, len(subset)):
            marginal_diversity += (np.linalg.norm(subset[i] - subset[j])-mean_embedding)/std_embedding
    marginal_diversity /= len(subset)
    return marginal_diversity
    
def statEmbedding(embeddings):
    distances = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            distance = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append(distance)
    distances = np.array(distances)
    mean_embedding = np.mean(distances)
    std_embedding = np.std(distances)
    return mean_embedding, std_embedding

# def getMPR(indices, labels, oracle, k, m):
#     # c(x)
#     weighting_clf = oracle_function(indices, labels, model=oracle)
#     # c(x) evaluated on data points and normalized
#     weighting_vector  = weighting_clf.predict(labels)
#     weighting_vector/= np.linalg.norm(weighting_vector)
#     rep = np.sum((1/k)*indices* weighting_vector -(1/m)*weighting_vector) # weighting_vector is c
#     return rep

# for clipclip benchmark method
def calc_feature_MI(features, labels, n_neighbors = 10, rs=1):
    return fs.mutual_info_classif(features, labels, discrete_features=False, copy=True, n_neighbors=n_neighbors, random_state=rs)

def return_feature_MI_order(features, data, sensitive_attributes, n_neighbors = 10, rs=1):
    labels_arr = np.reshape(data[:,sensitive_attributes], (-1, len(sensitive_attributes))) # enable intersectional groups
    labels = np.array([' '.join(map(str, row)) for row in labels_arr])
    feature_MI = calc_feature_MI(features, labels, n_neighbors, rs)
    feature_order = np.argsort(feature_MI)[::-1]
    return feature_order

def get_lower_upper_bounds(k, similarity_scores, labels, curation_labels):
    if curation_labels is None:
        curation_labels = labels
    a = cp.Variable(labels.shape[0])
    curation_mean = np.mean(curation_labels, axis=0)
    
    objective = cp.Maximize(similarity_scores.T @ a)
    constraints = [(labels.T @ a)/k == curation_mean, sum(a)==k, 0<=a, a<=1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    lb_opt = objective.value

    lb_indices = a.value


    #  find global optimal
    objective = cp.Maximize(similarity_scores.T @ a)
    constraints = [sum(a)==k, 0<=a, a<=1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    global_opt = objective.value

    ub_indices = a.value

    return lb_indices, ub_indices

# for PBM
def fon(l):
    try:
        return l[0]
    except:
        return None