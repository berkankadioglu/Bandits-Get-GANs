import numpy as np


class Bandit:
    """
    K-armed bandit in its simplest form.
    """
    def __init__(self, n_arms, stat_reward, conf_bound, eps, step_size):
        '''
        :param n_arms: number of MAB arms
        :param stat_reward: if True, use sample average. otw. constant step size
        :param conf_bound: if True, use upper confidence bound
        :param eps: exploration percentage. Only affects if stat_reward is false
        '''
        # Number of times each arm is pulled
        self.N = np.zeros(n_arms)
        # Action value estimates
        self.Q = np.zeros(n_arms)

        self.n_arms = n_arms
        self.stat_reward = stat_reward
        self.conf_bound = conf_bound
        self.eps = eps
        self.step_size = step_size
        self.last_pulled_arm = -1

    def update(self, reward):
        self.N[self.last_pulled_arm-1] += 1
        if self.stat_reward:  # sample average
            self.Q[self.last_pulled_arm-1] += (reward - self.Q[self.last_pulled_arm-1])/self.N[self.last_pulled_arm-1]
        else:  # constant step size
            self.Q[self.last_pulled_arm - 1] += (reward - self.Q[self.last_pulled_arm - 1]) * self.step_size

    def choose_arm(self):
        # k is in range [1,n_arms]
        temp = np.random.rand()
        if temp < self.eps:
            pulled_arm = np.random.permutation(range(self.n_arms))[0] + 1
        else:
            if self.conf_bound and np.all(self.N != 0):
                pulled_arm = np.argmax(self.Q + 1*np.sqrt(np.log(np.sum(self.N))/self.N)) + 1
            else:
                pulled_arm = np.argmax(self.Q) + 1

        self.last_pulled_arm = pulled_arm
        return pulled_arm


if __name__ == '__main__':

    bandito = Bandit(n_arms=5, eps=.1)

    # Start your epoch loop here.

    k = bandito.choose_arm()

    # you got k. train with mini batches accordingly, record new loss:
    loss_g = 5
    loss_d = 15/k  # This is what we defined.
    total_loss = loss_g + loss_d

    bandito.update(reward=-total_loss)
    k = bandito.choose_arm()


