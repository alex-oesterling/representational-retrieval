import numpy as np 
from sklearn.linear_model import LinearRegression
import sklearn.feature_selection as fs

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
    if model is not None and model == "linearregressiontheoretical":
        if curation_set is not None:
            m = curation_set.shape[0]
            term1 = 1/(k**2) * np.sum(np.outer(dataset, dataset.T))
            term2 = 1/(k*m) * np.sum(np.outer(dataset, curation_set.T))
            term3 = 1/(m**2) * np.sum(np.outer(curation_set, curation_set.T))
            mpr = term1+term2+term3
        else:
            m = dataset.shape[0]
            term1 = 1/(k**2) * np.sum(np.outer(dataset, dataset.T))
            term2 = 1/(k*m) * np.sum(np.outer(dataset, dataset.T))
            term3 = 1/(m**2) * np.sum(np.outer(dataset, dataset.T))
            mpr = term1+term2+term3
        return mpr

    reg = oracle_function(indices, dataset, curation_set=curation_set, model=model)

    if curation_set is not None:
        expanded_dataset = np.concatenate((dataset, curation_set), axis=0)
        m = curation_set.shape[0]
        c = reg.predict(expanded_dataset)
        c /= np.linalg.norm(c)
        mpr = np.abs(np.sum((indices/k)*c[:dataset.shape[0]]) - np.sum((1/m)*c[dataset.shape[0]:]))
    else:
        m = dataset.shape[0]
        c = reg.predict(dataset)
        c /= np.linalg.norm(c)
        mpr = np.abs(np.sum((indices/k)*c) - np.sum((1/m)*c))
    
    return mpr

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

# for PBM
def fon(l):
    try:
        return l[0]
    except:
        return None