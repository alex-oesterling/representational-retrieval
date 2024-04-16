import numpy as np
import cvxpy as cp
from sklearn.linear_model import LinearRegression

def sup_function(indices, dataset, model=LinearRegression):
    m = dataset.shape[0]
    k = int(np.sum(indices))
    alpha = (indices/k - 1/m)
    reg = model().fit(dataset, alpha)
    return reg.predict(dataset)

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
    
## @Carol: Please implement MMR here! ideally taking in your label matrix, similarity vector, and a lambda, get the set of retrieved vectors and then return the representation cost and the similarity score.
class MMR():
    def __init__(self):
        return

class GreedyOracle():
    def __init__(self, similarity_scores, labels):
        self.m = similarity_scores.shape[0]
        self.d = labels.shape[1]

        self.a = cp.Variable(self.m)
        self.y = cp.Variable(self.d)
        self.rho = cp.Parameter(nonneg=True) #similarity value
        self.C = labels

        self.similarity_scores = similarity_scores
        self.labels=labels

    def fit(self, k, num_iter, rho):
        self.objective = cp.Maximize(self.similarity_scores.T @ self.a)
        self.constraints = [sum(self.a)==k, 0<=self.a, self.a<=1]
        self.prob = cp.Problem(self.objective, self.constraints)
        self.prob.solve(solver=cp.ECOS,warm_start=True)

        print(self.a.value)
        print(np.sum((1/k)*self.a.value-(1/self.m)))


        # reps = []
        # sims = []
        for _ in range(num_iter):
            c = self.sup_function(self.a.value, k)
            if np.sum((1/k)*self.a.value*c-(1/self.m)*c) < rho:
                print("constraints satisfied, exiting early")
                break
            # print(lam, flush=True)
            # c = lam*reg.predict(self.C)
            self.max_similarity(c, k, rho)

        rep = np.sum((1/k)*self.a.value*c-(1/self.m)*c)
        sim = self.a.value.T@self.similarity_scores

        sparsity = sum(self.a.value>1e-4)
        at = self.a.value
        at[np.argsort(at)[::-1][k:]] = 0
        at[at>1e-5] = 1.0 

        rounded_rep = np.sum((1/k)*at*c-(1/self.m)*c)
        rounded_sim = at.T@self.similarity_scores

        return self.a.value, rep, sim#, rounded_rep, rounded_sim, sparsity


    def max_similarity(self, c, k, rho):
        print(rho)
        print(np.sum((1/k)*self.a.value*c-(1/self.m)*c), flush=True)
        self.constraints.append(cp.sum((1/k)*cp.multiply(self.a, c)-(1/self.m)*c)<=rho)
        self.prob = cp.Problem(self.objective, self.constraints)
        self.prob.solve(solver=cp.ECOS,warm_start=True)

    def sup_function(self, a, k):
        alpha = (a/k - 1/self.m)
        reg = LinearRegression().fit(self.C, alpha)
        return reg.predict(self.C)