import numpy as np
import OnlineVariance as ov
class PositiveStrategy(object):
    """
    Positive strategy selector.

    Defaults:
    K=2 arms
    D=2 features per arm
    """
    def __init__(self, K=2, D=2):
        self.K = K
        self.D = D

        self.stats = np.empty((K,D), dtype=object)
        for k in range(0,K):
            for d in range(0,D):
                self.stats[k,d] = ov.OnlineVariance(ddof=0)

    """
    Get the arms*vectors matrix of mean estimates
    """
    def mu(self):
        result = np.zeros((self.K,self.D))
        for k in range(0,self.K):
            for d in range(0,self.D):
                result[k,d] = self.stats[k,d].mean
        return result

    """
    Get the arms*vectors matrix of standard deviation estimates
    """
    def sigma(self):
        result = np.zeros((self.K, self.D))
        for k in range(0,self.K):
            for d in range(0,self.D):
                result[k,d] = self.stats[k,d].std
        return result

    """
    Include data in the model.

    Takes as input an arm, a feature vector, and an observed value.

    Stats are incrementally updated for each observed (arm,feature) combination.

    Note that we model all (arm,feature) combinations independently. Covariance
    is not considered.
    """
    def include(self, arm, features, value):
        for d in range(0,self.D):
            if features[d] > 0:
                self.stats[arm,d].include(value)

    """
    Estimate the expected value of an arm, given the current feature vector.

    This considers all previously observed outcomes added by calls to include()
    """
    def estimate(self, arm, features):
        return np.sum(features * list(map(lambda x: np.random.normal(x.mean, x.std if x.std > 0 else 1), self.stats[arm])))

    """
    Returns RMSE for all observed data, given a set of (presumably prior) weights
    """
    def rmse(self, weights):
        return np.sqrt(np.mean((weights - self.mu())**2)/self.K)
