import numpy as np
import OnlineVariance as ov
class Simulator(object):
    """
    Simulate model
    epsilon=0.05 learning rate
    """
    def __init__(self, model, epsilon=0.05):
        self.model = model
        self.K = model.K
        self.D = model.D
        self.epsilon = epsilon

        self.stats = np.empty((self.K,self.D), dtype=object)
        for k in range(0,self.K):
            for d in range(0,self.D):
                self.stats[k,d] = ov.OnlineVariance(ddof=0)

    def simulate(self,features,rewards,weights):
        N = rewards.size/self.K

        regret = np.zeros((N,1))
        rmse = np.zeros((N,1))

        for i in range(0,N):
            F = features[i]
            R = rewards[i]
   
            #our estimate and corresponding choice
            armMaxEstimate = 0.
            armChoice = 0

            #known reward and correct choice
            armMaxReward = 0.
            armOptimal = 0

            for k in range(0,self.K):
                #identify the optimal arm to choose
                if R[k] > armMaxReward:
                    armMaxReward = R[k]
                    armOptimal = k

                #choose an arm with best estimate based on current model
                armEstimate = self.model.estimate(k,F)
                if armEstimate > armMaxEstimate:
                    armMaxEstimate = armEstimate
                    armChoice = k

            #learn from an arm other than best estimate with p=epsilon
            learn = np.random.uniform() <= self.epsilon
            if learn:
                armAlt = armChoice
                while (armAlt == armChoice):
                    armAlt = int(np.random.uniform() * self.K)
                armChoice = armAlt

            #calculate reward and regret for chosen arm
            armReward = R[armChoice]
            armRegret = armMaxReward - armReward
            regret[i] = armRegret
            rmse[i]   = self.model.rmse(weights)

            #reward/penalize accordingly
            if armRegret == 0:
                self.model.include(armChoice, F, armReward)
            else:
                self.model.include(armChoice, F, -1 * armRegret)
            
        return regret, rmse
