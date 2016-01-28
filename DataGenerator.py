import numpy as np

class DataGenerator():
    """
    Generate badit data.

    Defaults:
    K=2 arms
    D=2 features/arm
    """
    def __init__(self,K=2,D=2,feature_type='binary',reward_type='binary'):
        
        self.D = D # dimension of the feature vector
        self.K = K # number of bandits
        self.reward_type = reward_type
        self.feature_type = feature_type
        self.means = np.random.normal(size=self.K)
        self.stds = 1 + 2*np.random.rand(self.K)
        
        # generate the weight vectors.  initialize estimate of feature
        # importance for each arm's d features
        self.generate_weight_vectors()
        
    def generate_weight_vectors(self,loc=0.0,scale=1.0):
        self.W = np.random.normal(loc=loc,scale=scale,size=(self.K,self.D))
        #self.W = np.ones((self.K,self.D))

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

    # generate all bernoulli rewards ahead of time
    def generate_bernoulli_bandit_data(self,num_samples):
        #initialize parameter estimates
        CTRs_that_generated_data = np.tile(np.random.rand(self.K),(num_samples,1))
        #did the trial succeed?
        true_rewards = np.random.rand(num_samples,self.K) < CTRs_that_generated_data
        return true_rewards,CTRs_that_generated_data

    # Thompson Sampling
    # basic idea: samples from distribution and compares those values for the arms instead
    # http://www.economics.uci.edu/~ivan/asmb.874.pdf
    # http://camdp.com/blogs/multi-armed-bandits
    def thompson_sampling(self,observed_data):
        return np.argmax( np.random.beta(observed_data[:,0], observed_data[:,1]) )


    # the bandit algorithm
    def run_bandit_alg(self,true_rewards,CTRs_that_generated_data,choice_func):
        num_samples,K = true_rewards.shape
        observed_data = np.zeros((K,2))
        # seed the estimated params
        prior_a = 1. # aka successes
        prior_b = 1. # aka failures
        observed_data[:,0] += prior_a # allocating the initial conditions
        observed_data[:,1] += prior_b
        regret = np.zeros(num_samples)

        for i in range(0,num_samples):
            # pulling a lever & updating observed_data
            this_choice = choice_func(observed_data)

            # update parameters
            if true_rewards[i,this_choice] == 1:
                update_ind = 0
            else:
                update_ind = 1

            observed_data[this_choice,update_ind] += 1

            # updated expected regret
            regret[i] = np.max(CTRs_that_generated_data[i,:]) - CTRs_that_generated_data[i,this_choice]

        cum_regret = np.cumsum(regret)

        return cum_regret
