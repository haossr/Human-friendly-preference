import numpy as np

class LinearRandomDataset:
    def __init__(self, N=100, F=8, sigma=0, seed=None, utility_fn=None):
        self._N = N
        self._F = F
        if utility_fn is None:
            self.beta = np.random.rand(self._F)
            self._U = lambda x: np.matmul(x, self.beta)
        else:
            self._U = utility_fn
        self._sigma = sigma
        self._seed = seed
        self._reset()
        
    def _reset(self):
        if self._seed is not None:
            np.random.seed(self._seed)
        self.X = np.random.rand(self._N, self._F)
        self.U = np.array([self._U(x) for x in self.X])
    
    def sample_full(self, M):
        eta = np.random.rand(M)

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

    def get_groundtruth(self):
        return {"U": self.U,
                "X": self.X,
                "beta": self.beta}



