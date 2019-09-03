import numpy as np

class DataGenerator():
    """
    Generate bandit data.

    Defaults:
    K = number of arms (default 2)
    D = number of features per arm (default 2)
    reward_type = one of: binary, positive, mixed
    """
    def __init__(self,K=2,D=2,feature_type='binary',reward_type='binary'):
        
        self.D = D # dimension of the feature vector
        self.K = K # number of bandits
        self.reward_type = reward_type
        self.feature_type = feature_type
        #initialize vectors of size K...
        #...representing the mean expected payouts for the arms
        self.means = np.random.normal(size=self.K)
        #...and the corresponding standard deviations of the expected payouts for the arms
        self.stds = 1 + 2*np.random.rand(self.K)
        
        self.generate_weight_vectors()
        
    """
    Generate weight matrix. Initialized randomly.

    For K arms and D features, creae a matrix of size (K,D).
    Any given cell (Ki,Dj) represents the magnitude that feature
    Dj has on the probability that arm Ki is the optimal choice.
    """
    def generate_weight_vectors(self,loc=0.0,scale=1.0):
        self.W = np.random.normal(loc=loc,scale=scale,size=(self.K,self.D))
        #self.W = np.ones((self.K,self.D))

    """
    Generate N observations.

    Returns two arrays:
    - X: N feature vectors of size D for the observation
    - R: N reward vectors of size K corresponding to observations X

    Note that this is a *data generator* and has complete information
    about the reward of each arm in K. These data are used by the simulator
    to choose its best guess of a single arm, then we can calculate the
    amount of "regret" (magnitude of lost opportunity) if a suboptimal choice
    was made.
    """
    def generate_samples(self,n=1000):
        # the sample feature vectors X are only binary

        if self.feature_type == 'binary':
            X = np.random.randint(0,2,size=(n,self.D))
        elif self.feature_type == 'integer':
            X = np.random.randint(0,5,size=(n,self.D))
        
        # the rewards are functions of the inner products of the
        # feature vectors with (current) weight estimates       
        IP = np.dot(X,self.W.T)

        # now get the rewards
        if self.reward_type == 'binary':
            R = ((np.sign(np.random.normal(self.means + IP,self.stds)) + 1) / 2).astype(int)
        elif self.reward_type == 'positive':
            #R = np.random.lognormal(self.means + IP,self.stds)
            R = np.abs(np.random.normal(self.means + IP,self.stds))
        elif self.reward_type == 'mixed':
            R = (np.sign(np.random.normal(self.means + IP,self.stds)) + 1) / 2
            R *= np.random.lognormal(self.means + IP,self.stds)

        return X,R
