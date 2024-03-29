import numpy as np, math
from tqdm import tqdm

class Thompson:
    name = "Thompson"
    def __init__(self, bandits):
        self.num = len(bandits)
        self.bandits = bandits
        self.history = [[1] for _ in range(self.num)]

    def pull(self):
        bandit = max(range(self.num), key=lambda n:
        np.random.normal(np.mean(self.history[n]), np.std(self.history[n])))

        reward = self.bandits[bandit].pull()
        self.history[bandit].append(reward)
        return self.bandits[bandit], reward

class Thompson2:
    name = "Thompson2"
    def __init__(self, bandits):
        self.num = len(bandits)
        self.bandits = bandits
        self.prior, self.priormu = 100, 1
        self.sums, self.sig, self.times, self.mu = \
        [0] * self.num, [self.prior]*self.num, [0] * self.num, [self.priormu] * self.num
    

    def pull(self):
        bandit = max(range(self.num), key=lambda n:
        np.random.normal(self.mu[n], np.sqrt(self.sig[n])))
        reward = self.bandits[bandit].pull()
        self.sums[bandit] += reward
        self.times[bandit] += 1
        self.mu[bandit] = (self.mu[bandit]/self.sig[bandit]+self.sums[bandit]/.01)/\
        (1/self.sig[bandit]+self.times[bandit]/.01)
        self.sig[bandit] = 1/(1/self.prior+100*self.times[bandit])
        
        return self.bandits[bandit], reward

class Parameters:
    def __init__(self):
        self.n = 0
        self.x = []
        self.a = 1
        self.b = 1
        self.mu = 1
        self.sum = 0
    
    def update(self, x):
        self.n += 1
        self.sum += x
        self.x.append(x)
        self.mu = self.sum/self.n
        v = self.n

        self.a += 1/2
        self.b += ((v/(v + 1)) * (((x - self.mu)**2)/2))

    def __str__(self) -> str:
        return f'Alpha: {self.a}  Beta: {self.b}  Mu: {self.mu}  v: {self.n} est : {np.mean([np.random.gamma(self.a, 1/self.b) for _ in range(10)])}'
        


class Thompson3:
    name = "Thompson3"
    def __init__(self, bandits):
        self.num = len(bandits)
        self.bandits = bandits
        self.data = [Parameters() for _ in range(self.num)]
    def pull(self):
        bandit = max(range(self.num), key=lambda n:
            np.random.normal(self.data[n].mu, np.sqrt(1/np.random.gamma(self.data[n].a, 1/self.data[n].b))))
        r = self.bandits[bandit].pull()
        self.data[bandit].update(r)
        return self.bandits[bandit], r

class Random:
    name = "Random"
    def __init__(self, bandits):
        self.num = len(bandits)
        self.bandits = bandits
    def pull(self):
        bandit = np.random.choice(self.bandits)
        return bandit, bandit.pull()

class Greedy:
    name = "Greedy"
    def __init__(self, bandits):
        self.t = 1
        self.bandits = bandits
        self.num = len(bandits)
        self.totals = [0] * self.num
        self.times = [0] * self.num
        
    def pull(self):
        prob = 0.1 / math.log(self.t + 1.00001)
        bandit = np.random.randint(0, self.num-1) if np.random.random() < prob else\
                max(range(self.num), key=lambda n: self.totals[n]/max(self.times[n], 1))
        r = self.bandits[bandit].pull()
        self.times[bandit] += 1
        self.totals[bandit] += r
        self.t += 1
        return self.bandits[bandit], r

class UCB:
    name = "UCB"
    def __init__(self, bandits):
        self.c = 1
        self.t = 0
        self.bandits = bandits
        self.num = len(bandits)
        self.totals = [0] * self.num
        self.times = [0] * self.num

    def pull(self):
        bandit = max(range(self.num), key=lambda n: self.totals[n]/max(self.times[n], 1)+UCB.c*
        math.sqrt(math.log(max(self.t, 1))/max(self.times[n], 1)))
        r = self.bandits[bandit].pull()
        self.times[bandit] += 1
        self.totals[bandit] += r
        self.t += 1
        return self.bandits[bandit], r
    
    



class Bandit:
    def __init__(self, mean=0.5, std=0.1):
        self.mean = mean
        self.std = std

    def pull(self):
        return np.random.normal(self.mean, self.std)

    def __str__(self):
        return f"Mean: {self.mean}, std: {self.std}"