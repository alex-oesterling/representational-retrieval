import numpy as np
import cvxpy as cp
from sklearn.linear_model import LinearRegression

def oracle_function(indices, dataset, model=LinearRegression):
    m = dataset.shape[0]
    k = int(np.sum(indices))
    alpha = (indices/k - 1/m)
    reg = model().fit(dataset, alpha)
    return reg

class CVXPY():
    def __init__(self, similarity_scores, labels):
        m = similarity_scores.shape[0]
        d = labels.shape[1]
        self.similarity_scores = similarity_scores
        self.a = cp.Variable(m)
        self.y = cp.Variable(d)
        self.rho = cp.Parameter(nonneg=True) #similarity value
        self.C = labels
        self.cbar = self.C.mean(axis=0)
        self.objective = None
        self.constraints = None
        self.problem = None
    
    def get_lower_upper_bounds(self, k):
        objective = cp.Maximize(self.similarity_scores.T @ self.a)
        constraints = [(self.C.T @ self.a)/k == self.cbar, sum(self.a)==k, 0<=self.a, self.a<=1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        lb_opt = objective.value


        #  find global optimal
        objective = cp.Maximize(self.similarity_scores.T @ self.a)
        constraints = [(self.C.T @ self.a)/k == self.y, sum(self.a)==k, 0<=self.a, self.a<=1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        global_opt = objective.value

        return lb_opt, global_opt
    
    def fit(self, k, p):
        self.rho.value = p
        
        if self.objective is None:
            self.objective = cp.Minimize(cp.sum_squares(self.y-self.cbar))
        if self.constraints is None:
            self.constraints = [(self.C.T @ self.a)/k == self.y, sum(self.a)==k, 0<=self.a, self.a<=1,  self.similarity_scores.T @ self.a==self.rho]
        if self.problem is None:
            self.problem = cp.Problem(self.objective, self.constraints)

        self.problem.solve(solver=cp.ECOS,warm_start=True)

        at = self.a.value

        return at

        similarity = self.similarity_scores.T@at
        representation = np.sqrt(np.sum(np.power((self.C.T @ at)/k - self.cbar, 2)))

        return representation, similarity

    def fit_select(self, k, p):
        self.rho.value = p

        if self.objective is None:
            self.objective = cp.Minimize(cp.sum_squares(self.y-self.cbar))
        if self.constraints is None:
            self.constraints = [(self.C.T @ self.a)/k == self.y, sum(self.a)==k, 0<=self.a, self.a<=1,  self.similarity_scores.T @ self.a==self.rho]
        if self.problem is None:
            self.problem = cp.Problem(self.objective, self.constraints)

        self.problem.solve(solver=cp.ECOS,warm_start=True)

        sparsity = sum(self.a.value>1e-4)
        at = self.a.value
        at[np.argsort(at)[::-1][k:]] = 0
        at[at>1e-5] = 1.0 
        # representation = np.sqrt(np.sum(np.power((self.C.T @ at)/k - self.cbar, 2)))
        # similarity = self.similarity_scores.T @ at

        return at, sparsity

        return representation, similarity, sparsity
    
    def get_representation(self, indices, k):
        rep = np.sqrt(np.sum(np.power((self.C.T @ indices)/k - self.cbar, 2)))
        return rep
    
    def get_similarity(self, indices):
        sim = self.similarity_scores.T @ indices
        return sim
    
# MMR algorithm defined in PATHS using CLIP embedding
class MMR():
    def __init__(self, similarity_scores, labels, features):
        self.m = similarity_scores.shape[0]
        self.similarity_scores = similarity_scores
        # define what embedding to use for the diversity metric.
        # None (img itself), CLIP, or CLIP+PCA
        self.embeddings = features # the entire dataset (also used for retrieval), or a separate curation set
        self.mean_embedding = None
        self.std_embedding = None

    def fit(self, k, lambda_):
        if self.mean_embedding is None or self.std_embedding is None:
            self.statEmbedding()

        indices = np.zeros(self.m)
        selection = []
        for i in range(k):
            MMR_temp = np.full(len(self.embeddings), -np.inf)
            if i==0:
                idx = np.argmax(self.similarity_scores)
                selection.append(idx)
                indices[idx] = 1
                continue
            for j in range(len(self.embeddings)):
                if indices[j] == 1:
                    continue
                # temporary select the jth element
                indices[j] = 1
                MMR_temp[j] = (1-lambda_)* self.similarity_scores.T @ indices + lambda_ * self.marginal_diversity_score(indices,j)
                indices[j] = 0
            # select the element with the highest MMR 
            idx = np.argmax(MMR_temp)
            selection.append(idx)
            indices[np.argmax(MMR_temp)] = 1
        AssertionError(np.sum(indices)==k)
        MMR_cost = self.marginal_diversity_score(indices)
        return indices, MMR_cost, selection
    
    def statEmbedding(self):
        distances = []
        for i in range(len(self.embeddings)):
            for j in range(i+1, len(self.embeddings)):
                distance = np.linalg.norm(self.embeddings[i] - self.embeddings[j])
                distances.append(distance)
        distances = np.array(distances)
        self.mean_embedding = np.mean(distances)
        self.std_embedding = np.std(distances)


    def marginal_diversity_score(self, indices, addition_index=None):
        # compute the diversity score of the entire set
        if addition_index is None:
            marginal_diversity = 0
            subset = self.embeddings[indices==1]
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    marginal_diversity += (np.linalg.norm(subset[i] - subset[j])-self.mean_embedding)/self.std_embedding
            marginal_diversity /= len(subset)

        # compute the diversity score of an additional item
        else:
            # compute the diversity score of adding the addition_index to the current set
            marginal_diversity = 0
            indices[addition_index] = 0
            subset = self.embeddings[indices==1]
            for i in range(len(subset)):
                marginal_diversity += (np.linalg.norm(self.embeddings[addition_index] - subset[i])-self.mean_embedding)/self.std_embedding
            marginal_diversity /= len(subset)
        return marginal_diversity
        

# minimize MSE of a regression problem
class GreedyOracle():
    def __init__(self, similarity_scores, labels, model=LinearRegression):
        self.m = similarity_scores.shape[0]
        self.d = labels.shape[1]

        self.a = cp.Variable(self.m)
        self.y = cp.Variable(self.d)
        self.rho = cp.Parameter(nonneg=True) #similarity value
        self.C = labels

        self.similarity_scores = similarity_scores

        self.model=model

    def fit(self, k, num_iter, rho):
        self.objective = cp.Maximize(self.similarity_scores.T @ self.a)
        self.constraints = [sum(self.a)==k, 0<=self.a, self.a<=1]
        self.prob = cp.Problem(self.objective, self.constraints)
        self.prob.solve(solver=cp.ECOS,warm_start=True)

        print(self.a.value, flush=True)
        print(np.abs(np.sum((1/k)*self.a.value-(1/self.m))), flush=True)

        # reps = []
        # sims = []
        for _ in range(num_iter):
            reg = self.sup_function(self.a.value, k)
            c = reg.predict(self.C)
            c /= np.linalg.norm(c)
            if np.abs(np.sum((1/k)*self.a.value*c-(1/self.m)*c)) < rho:
                print("constraints satisfied, exiting early")
                print(np.abs(np.sum((1/k)*self.a.value*c-(1/self.m)*c)))
                print(rho)
                break
            self.max_similarity(c, k, rho)

        return self.a.value


    def max_similarity(self, c, k, rho):
        self.constraints.append(cp.abs(cp.sum((1/k)*cp.multiply(self.a, c)-(1/self.m)*c))<=rho)
        self.prob = cp.Problem(self.objective, self.constraints)
        self.prob.solve(solver=cp.ECOS,warm_start=True)

    def sup_function(self, a, k):
        alpha = (a/k - 1/self.m)
        reg = self.model().fit(self.C, alpha)
        return reg
    
    def get_representation(self, indices, k):
        reg = self.sup_function(indices, k)
        c = reg.predict(self.C)
        c /= np.linalg.norm(c)
        rep = np.abs(np.sum((1/k)*indices*c-(1/self.m)*c))
        return rep

    def get_similarity(self, indices):
        sim = indices.T@self.similarity_scores
        return sim