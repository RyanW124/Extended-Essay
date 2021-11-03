import numpy as np, math

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
    c = 1
    name = "UCB"
    def __init__(self, bandits):
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