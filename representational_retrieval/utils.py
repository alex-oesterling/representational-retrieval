import numpy as np 

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