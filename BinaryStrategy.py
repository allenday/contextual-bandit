import numpy as np
class BinaryStrategy(object):
    """
    Binary strategy selector.

    Defaults:
    K=2 arms
    D=2 features/arm
    epsilon=0.05 learning rate
    """
    def __init__(self, K=2, D=2, epsilon=0.05):
        self.K = K
        self.D = D
        self.epsilon = epsilon
        self.alpha = np.ones((K,D)).astype(int)
        self.beta  = np.ones((K,D)).astype(int)

#    @property
#    def std(self):
#        return np.sqrt(self.variance)

    def simulate(self,features,rewards,weights):
        N = rewards.size/self.K

        regret = np.zeros((N,1))
        rmse = np.zeros((N,1))

        for i in range(0,N):
            S = np.zeros((self.K,self.D))
            F = features[i]
            R = rewards[i]
    
            armOpt = 0
            armMax = 0.

            for k in range(0,self.K):
                armSum = 0.0
                for d in range(0,self.D):
                    alphaSample = self.alpha[k,d]
                    betaSample = self.beta[k,d]
                    s = np.random.beta(alphaSample,betaSample)
                    S[k,d] = s
                    part = s * F[d]
                armSum += part
                if armSum > armMax:
                    armMax = armSum
                    armOpt = k

            invest = np.random.uniform() <= self.epsilon
            #choose an arm to learn with p=epsilon
            if invest:
                armAlt = armOpt
                while (armAlt == armOpt):
                    armAlt = int(np.random.uniform() * self.K)
                armOpt = armAlt

            armReward = R[armOpt]
            armRegret = armMax - armReward
            regret[i] = abs(armRegret)
            rmse[i]  = np.sqrt(np.mean((weights - S)**2))

            if armReward > 0:
                self.alpha[armOpt] += F
            else:
                self.beta[armOpt] += F
            
        return regret, rmse
