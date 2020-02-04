import numpy as np

class Exp:
    def __init__(self, N=100, F=8, M=40, sigma=0, seed=123):
        self.N = N
        self.F = F
        self.M = M
        self.sigma = sigma
        self.seed = seed
        self.sample()
        
    def sample(self):
        np.random.seed(123)
        self.theta = np.random.rand(self.F)
        self.eta = np.random.rand(self.M)
        self.X = np.random.rand(self.N, self.F)
        self.C_ind = np.random.randint(self.N, size=(self.M, 2))
        self.C_x = self.X[self.C_ind]
        
        self.C_y_raw = self.U(self.C_x[:,0,:]) -  self.U(self.C_x[:,1,:]) \
        + self.eta * self.sigma
        self.C_y = (self.C_y_raw > 0).astype(np.int)
        return self.gen_data()
        
    def U(self, X):
        return np.matmul(X, self.theta)
    
    def ground_truth(self):
        U = self.U(self.X)
        umax = np.max(U) 
        umax_ind = np.argmax(U) 
        return {'umax': umax,
                'umax_ind': umax_ind,
                'theta': self.theta}
    
    def gen_data(self):
        return {'N': self.N,
                'F': self.F,
                'M': self.M,
                'X': self.X,
                'C_x1': self.C_x[:,0,:],
                'C_x2': self.C_x[:,1,:],
                'C_y': self.C_y,
                'sigma': self.sigma}
