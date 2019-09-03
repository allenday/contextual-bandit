import numpy as np
import OnlineVariance as ov
"""
TODO verbose description of class
"""
class Simulator(object):
    """
    Simulate model
    epsilon=0.05 learning rate
    TODO constructor description
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

    """
    Simulate results. Takes as input:

    - `features`: observed list of feature vectors
    - `rewards`: observed list of rewards, corresponding to the feature vectors
    - `weights`: known true weights of each feature, for each arm. used to calculate regret and rmse

    TODO describe output
    """
    def simulate(self,features,rewards,weights):
        #figure out how many observations there are
        N = int(rewards.size/self.K)

        #initialize regret, rmse
        regret = np.zeros((N,1))
        rmse = np.zeros((N,1))

        for i in range(0,N):
            F = features[i]
            R = rewards[i]


            #identify the TRUE OPTIMAL arm to choose
            armOptimal = np.argmax(R)
            armMaxReward = R[armOptimal]

            #choose the best ESTIMATED OPTIMAL arm based on current model
            armChoices = list(map(lambda x: self.model.estimate(x,F), range(0,self.K)))
            armChoice = np.argmax(armChoices)
            armMaxEstimate = armChoices[armChoice]
   
            #choose an ESTIMATED NON-OPTIMAL arm for the purpose of learning with p=epsilon
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
