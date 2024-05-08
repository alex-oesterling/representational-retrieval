import torch
    
class ADMM():
    def __init__(self, rho=1., retrieval_penalty=0.2, tol=1e-6, max_iter=10000, device="cuda", verbose=False):
        self.rho = rho
        self.retrieval_penalty = retrieval_penalty
        self.tol = tol
        self.max_iter = max_iter
        self.device = device
        self.verbose = verbose

    def step(self, Cb, Q_cho, a_star, u, k):
        an = torch.cholesky_solve(Cb + self.rho*(a_star - u), Q_cho)
        #a_starn = #FIXME #torch.where((an + u - self.retrieval_penalty/self.rho) > 0, an + u - self.retrieval_penalty/self.rho, 0)
        a_starn = 0 #TBD!!
        ## project onto hyperplane
        ones = torch.ones(an.size).to(self.device)
        proj_hyperplane = an-(torch.dot(ones, an)-k)/(torch.linalg.norm(ones)**2)*ones
        un = u + an - a_starn
        return an, a_starn, un
    
    def fit(self, C, s, k):
        '''
        Fit using ADMM
        C: m x d "attribute matrix" of various features/sensitive attributes per sample
        s: an m x 1 vector of similarity scores for each sample with our query.
        k: the number of items to retrieve
        '''
        ## iterates are dim m, the number of samples
        m = C.shape[0]

        ## size: c x c
        Q = (2/(k**2)) *C @ C.T + (torch.eye(m)*self.rho).to(self.device)
        Cb = (2/(k*m))*(C@C.T@torch.ones((m, 1)) + self.retrieval_penalty*k*m*s)
        Cb = Cb.to(self.device)

        # factor Q for quicker solve -- this is critical.
        Q_cho = torch.linalg.cholesky(Q)

        # iterates, size: c x batch
        a = torch.randn((m, 1)).to(self.device)
        a_star = torch.randn((m, 1)).to(self.device)
        u = torch.randn((m, 1)).to(self.device)

        for ix in range(self.max_iter):
            a_star_old = a_star

            a, a_star, u = self.step(Cb, Q_cho, a_star, u, k)

            res_prim = torch.linalg.norm(a-a_star, dim=0)
            res_dual = torch.linalg.norm(self.rho*(a_star-a_star_old), dim=0)

            if (res_prim.max() < self.tol) and (res_dual.max() < self.tol):
                break
        if self.verbose:
            print("Stopping at iteration {}".format(ix))
            print("Prime Residual, r_k: {}".format(res_prim.mean()))
            print("Dual Residual, s_k: {}".format(res_dual.mean()))
        return a_star.T