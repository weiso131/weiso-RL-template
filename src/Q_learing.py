import numpy as np

class Q_learning():
    def __init__(self, observationSize : tuple, observationBound : tuple, actionSpace : list):
        self.Q_form = np.zeros(observationSize + (len(actionSpace), ))
        self.actionSpace = actionSpace
        self.observationSize = observationSize
        self.observationBound = observationBound
        self.actionIndex = {}
        for i in range(len(actionSpace)):
            self.actionIndex[actionSpace[i]] = i

    def step(self, state : tuple, Lambda=0):
        if (np.random.rand() > Lambda):
            return self.actionSpace[np.argmax(self.Q_form[self.discrete(state)])]
        else:
            return np.random.choice(self.actionSpace)
        
    def update(self, oldState : tuple, newState : tuple, \
               action, reward, lr=0.01, gamma=0.99):
        self.Q_form[self.discrete(oldState) + (self.actionIndex[action],)] = \
        (1 - lr) * self.Q_form[self.discrete(oldState) + (self.actionIndex[action],)] +\
        lr * (reward + gamma * np.max(self.Q_form[self.discrete(newState)]))

    def discrete(self, state : np.ndarray)->tuple:
        discreteState = []
        
        for i in range(len(self.observationSize)):
            if (self.observationSize[i] == 1):
                discreteState.append(0)
                continue
            obs = max(min(state[i], self.observationBound[i][1]), self.observationBound[i][0]) - self.observationBound[i][0]
            bucket = (self.observationBound[i][1] - self.observationBound[i][0]) / self.observationSize[i]
            discreteState.append(int(obs/bucket) - 1)
        
        return tuple(discreteState)



    