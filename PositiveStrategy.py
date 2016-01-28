import numpy as np
import OnlineVariance as ov
class PositiveStrategy(object):
    """
    Positive strategy selector.

    Defaults:
    K=2 arms
    D=2 features/arm
    epsilon=0.05 learning rate
    """
    def __init__(self, K=2, D=2, epsilon=0.05):
        self.K = K
        self.D = D
        self.epsilon = epsilon

        self.stats = np.empty((K,D), dtype=object)
        for k in range(0,K):
            for d in range(0,D):
                self.stats[k,d] = ov.OnlineVariance(ddof=0)

    def mu(self):
        result = np.zeros((self.K,self.D))
        for k in range(0,self.K):
            for d in range(0,self.D):
                result[k,d] = self.stats[k,d].mean
        return result

    def sigma(self):
        result = np.zeros((self.K, self.D))
        for k in range(0,self.K):
            for d in range(0,self.D):
                result[k,d] = self.stats[k,d].std
        return result

    def include(self, arm, features, value):
        for d in range(0,self.D):
            if features[d] > 0:
                self.stats[arm,d].include(value)

    def estimate(self, arm, features):
        return np.sum(features * map(lambda x: np.random.normal(x.mean, x.std if x.std > 0 else 1), self.stats[arm]))

    def rmse(self, weights):
#        print weights
#        print self.mu()
#        print weights - self.mu()
#        print np.mean((weights - self.mu())**2)
        return np.sqrt(np.mean((weights - self.mu())**2)/self.K)
