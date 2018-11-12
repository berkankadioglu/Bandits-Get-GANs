import numpy as np


class Bandit:
    """
    K-armed bandit in its simplest form.
    """
    def __init__(self, n_arms, eps):
        # Number of times each arm is pulled
        self.N = np.zeros(n_arms)
        # Action value estimates
        self.Q = np.zeros(n_arms)
        # Epsilon
        self.eps = eps

        self.n_arms = n_arms
        self.last_pulled_arm = -1

    def update(self, reward):
        self.N[self.last_pulled_arm] += 1
        self.Q[self.last_pulled_arm] += (reward - self.Q[self.last_pulled_arm])/self.N[self.last_pulled_arm]

    def choose_arm(self):

        temp = np.random.rand()
        if temp < self.eps:
            pulled_arm = np.random.permutation(range(self.n_arms))[0]
        else:
            pulled_arm = np.argmax(self.Q)[0]

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


