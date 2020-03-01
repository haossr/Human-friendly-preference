import numpy as np

class LinearRandomDataset:
    def __init__(self, N=100, F=8, sigma=0, seed=None):
        self._N = N
        self._F = F
        self._sigma = sigma
        self._seed = seed
        self._reset()
        
    def _reset(self):
        self.beta = np.random.rand(self._F)
        if self._seed is not None:
            np.random.seed(self._seed)
        self.X = np.random.rand(self._N, self._F)
        self.U = np.array([self._U(x) for x in self.X])
   
    def _U(self, X): 
        return np.matmul(X, self.beta)
    
    def _U_partial(self, X, D):
        return np.matmul(X[...,D], self.beta[D])

    def sample_full(self, M):
        eta = np.random.randn(M)
        C_ind = np.random.randint(self._N, size=(M, 2))
        C_x = self.X[C_ind] 
        C_y_raw = self._U(C_x[:,0,:]) -  self._U(C_x[:,1,:]) \
        + eta * self._sigma
        C_y = (C_y_raw > 0).astype(np.int)
        
        return {'N': self._N,
                'F': self._F,
                'M': M,
                'X': self.X,
                'C_x1': C_x[:,0,:],
                'C_x2': C_x[:,1,:],
                'C_y': C_y,
                'sigma': self._sigma}

    def sample_partial(self, M, B):
        eta = np.random.randn(M, B+1) * np.sqrt(1/2)
        C_ind = np.random.randint(self._N, size=(M, B+1))

        C_X = self.X[C_ind] 
        C = np.zeros((M, B, 2, self._F))
        C_U = np.zeros((M, B+1))
        D = np.zeros((M, 2), dtype=np.int8)
        C_y = np.zeros((M, B), dtype=np.int8)
        for i in range(M):
            D[i,:] = np.random.choice(range(self._F), 2, replace=False)
            C_U[i,:] = self._U_partial(C_X[i], D[i,:]) + eta[i,:]
        max_ind = np.argmax(C_U, axis=1)
        for i in range(M):
            max_ind = np.argmax(C_U[i])
            offset = 0
            for j in range(B):
                if j == max_ind:
                   offset = 1
                if np.random.rand() > 0.5:
                    C[i,j,0,:] = C_X[i, max_ind]
                    C[i,j,1,:] = C_X[i, j+offset] 
                    C_y[i,j] = 1     
                else:
                    C[i,j,1,:] = C_X[i, max_ind] 
                    C[i,j,0,:] = C_X[i, j+offset] 
                    C_y[i,j] = 0     

        return {'N': self._N,
                'F': self._F,
                'M': M,
                'B': B,
                'X': self.X,
                'D': D+1,
                'C': C,
                'C_y': C_y,
                'sigma': self._sigma}

    def get_groundtruth(self):
        return {"U": self.U,
                "X": self.X,
                "beta": self.beta}


