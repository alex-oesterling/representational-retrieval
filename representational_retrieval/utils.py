import numpy as np 
from sklearn.linear_model import LinearRegression
import sklearn.feature_selection as fs

def oracle_function(indices, dataset, model=None):
    if model is None:
        model = LinearRegression()
    m = dataset.shape[0]
    k = int(np.sum(indices))
    alpha = (indices/k - 1/m)
    reg = model.fit(dataset, alpha)
    return reg

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

def getMPR(indices, labels, oracle, k, m):
    # c(x)
    weighting_clf = oracle_function(indices, labels, model=oracle)
    # c(x) evaluated on data points and normalized
    weighting_vector  = weighting_clf.predict(labels)
    weighting_vector/= np.linalg.norm(weighting_vector)
    rep = np.sum((1/k)*indices* weighting_vector -(1/m)*weighting_vector) # weighting_vector is c
    return rep

# for clipclip benchmark method
def calc_feature_MI(features, labels, n_neighbors = 10, rs=1):
    return fs.mutual_info_classif(features, labels, discrete_features=False, copy=True, n_neighbors=n_neighbors, random_state=rs)

def return_feature_MI_order(features, data, sensitive_attributes, n_neighbors = 10, rs=1):
    labels = data[sensitive_attributes].apply(lambda x: ' '.join(x), axis=1) 
    print(labels)
    feature_MI = calc_feature_MI(features, labels, n_neighbors, rs)
    print(feature_MI)
    feature_order = np.argsort(feature_MI)[::-1]
    print(feature_MI[feature_order])
    return 

# for PBM
def fon(l):
    try:
        return l[0]
    except:
        return None