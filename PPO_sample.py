from torch import nn
import torch.nn.functional as F

import gym
from src.PPO import PPO

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)  # 使用softmax確保輸出是概率分佈
class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 1)  # 僅一個輸出，表示狀態的價值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class MyPPO(PPO):
    def __init__(self, env, policyNetwork, valueNetwork):
        super().__init__(policyNetwork, valueNetwork)
        self.env = env


    def learn(self, timeStep=1000, dataNum = 4096, lr=0.003, episode=0.2, epoch=10, batchSize=256):
        print("start learning")
        for i in range(timeStep):
            playtime_count = 0
            survive_avg = 0
            print(f"time step:{i + 1}", end=" ")
            while (len(self.ExperienceHistory['oldstate']) < dataNum):
                state, _ = self.env.reset()
                done = False
                while (not done):
                    action = self.getAction(state)

                    next_state, reward, done, _, __ = self.env.step(action)
                    
                    self.ExperienceHistory['oldstate'].append(state)
                    self.ExperienceHistory['state'].append(next_state)
                    self.ExperienceHistory['action'].append(action)
                    self.ExperienceHistory['reward'].append(reward)
                    self.ExperienceHistory['done'].append(int(done))
                    state = next_state
                    survive_avg += 1
                    if done:
                        playtime_count += 1
                        break
            print("平均生存時間:", survive_avg / playtime_count)
            self.train(epochs=epoch, lr=lr, episode=episode, batch_size=batchSize)



env = gym.make('CartPole-v1')
policyNetwork = PolicyNetwork(4, 2)
valueNetwork = ValueNetwork(4)
agent = MyPPO(env, policyNetwork=policyNetwork, valueNetwork=valueNetwork)
agent.learn(lr=0.001, dataNum=4096)