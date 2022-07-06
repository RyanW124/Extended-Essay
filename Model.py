import numpy as np
from tqdm import tqdm

class Algorithm:
    def __init__(self, n, bandits=None, *, alg) -> None:
        self.n = n
        self.bandits = [Bandit() for _ in range(n)] if bandits is None else bandits
        self.params = [alg(bandit) for bandit in self.bandits]
    def draw(self, update=True):
        bandit_n = max(range(self.n), key=lambda index: self.params[index].draw())
        bandit = self.bandits[bandit_n]
        reward = bandit.draw()
        if update:
            self.params[bandit_n].update(reward)
        return bandit, reward

class TS(Algorithm):
    name = "TS"
    def __init__(self, n, bandits=None) -> None:
        super().__init__(n, bandits, alg=TSbandit)
class TS2(Algorithm):
    name = "TS2"
    def __init__(self, n, bandits=None) -> None:
        super().__init__(n, bandits, alg=TS2bandit)

class UCB(Algorithm):
    name = "UCB"
    def __init__(self, n, bandits=None) -> None:
        super().__init__(n, bandits, alg=UCBbandit)
        UCBbandit.t = 1
    def change_c(self, c):
        for i in self.params:
            i: UCBbandit
            i.c = c

    @classmethod
    def hyperparameter_tuning(self, n, num_sets=5):
        bandits = [[Bandit(i) for i in range(np.random.randint(50, 1000))] for _ in range(num_sets)]
        points = np.random.uniform(0, 2, n)
        max_reward, max_ucb = -float('inf'), None
        for _ in tqdm(range(n)):
            reward = 0
            for i in range(num_sets):
                ucb = UCB(len(bandits[i]), bandits[i])
                ucb.change_c(points[i])
                for _ in range(ucb.n):
                    _, r = ucb.draw()
                    reward += r
            if reward > max_reward:
                max_reward = reward
                max_ucb = points[i]
        return max_ucb



class UCBbandit:
    t = 1
    def __init__(self, bandit) -> None:
        self.bandit = bandit
        self.sum = 0
        self.N = 0.0001
        self.c = 1
    def update(self, r):
        self.sum += r
        self.N += 1
        UCBbandit.t += 1
    def draw(self):
        return self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
    @property
    def Q(self):
        return self.sum / self.N
    
class TS2bandit:
    TAU = 1/25
    def __init__(self, bandit) -> None:
        self.bandit = bandit
        self.mu = 0
        self.tau = 0.0001
    def update(self, r):
        new_mu = (self.tau*self.mu+self.TAU*r)/(self.tau+self.TAU)
        new_tau = 1/(self.tau+self.TAU)

        self.mu = new_mu
        self.tau = new_tau
    def draw(self):
        return np.random.normal(self.mu, np.sqrt(1/self.tau))
class TSbandit:
    def __init__(self, bandit) -> None:
        self.bandit = bandit
        self.mu = 0
        self.alpha = 0.5
        self.beta = 1
        self.nu = 0
    def update(self, r):
        new_mu = (self.mu * self.nu + r)/ (self.nu + 1)
        new_nu = self.nu + 1
        new_alpha = self.alpha + 0.5
        new_beta = self.beta + self.nu * (r - self.mu) ** 2 / (2*(self.nu+1))
        self.mu, self.nu, self.alpha, self.beta = new_mu, new_nu, new_alpha, new_beta
    def draw(self):
        if self.nu == 0:
            return float('inf')
        variance = 1/np.random.gamma(self.alpha, 1/self.beta)
        return np.random.normal(self.mu, np.sqrt(variance))

class Bandit:
    N = 0
    def __init__(self, n, mean=None, std=None) -> None:
        self.mean = np.random.uniform(0, 100) if mean is None else mean
        self.std = np.random.uniform(0, 25) if std is None else std
        self.n = n
        self.N = n
    def draw(self):
        return np.random.normal(self.mean, self.std)
    def __str__(self) -> str:
        return f"Bandit {self.n}: mean {self.mean} std {self.std}"

