import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F

class PPO():
    def __init__(self, policyNetwork, valueNetwork):
        self.PolicyNetwork = policyNetwork
        self.ValueNetwork = valueNetwork
        self.ExperienceHistory = {}
        self.ExperienceHistory['oldstate'] = []
        self.ExperienceHistory['state'] = []
        self.ExperienceHistory['action'] = []
        self.ExperienceHistory['reward'] = []
        self.ExperienceHistory['done'] = []

    def getAction(self, state):
        if (type(state) != Tensor):
            state = Tensor(state)
        predict = self.PolicyNetwork(state)

        action_list = torch.distributions.Categorical(predict)

        action = action_list.sample().item()#根據機率隨機選一個動作

        return action

    
    def train(self, batch_size=64, epochs=10, gamma=0.9, lr=1e-3, episode=0.2, lmbda=0.95):
        # 檢查CUDA是否可用並設定device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 載入數據到DataLoader
        # 數據結構 [oldstate, state, action, reward]
        
        

        self.PolicyNetwork.to(device, dtype=torch.float32)
        self.ValueNetwork.to(device, dtype=torch.float32)
        # 定義優化器
        optimizer_policy = optim.Adam(self.PolicyNetwork.parameters(), lr=lr, eps=1e-9)
        optimizer_value = optim.Adam(self.ValueNetwork.parameters(), lr=lr, eps=1e-9)



        State = torch.tensor(self.ExperienceHistory['oldstate'])
        NextState = torch.tensor(self.ExperienceHistory['state'])
        Action = torch.tensor(self.ExperienceHistory['action']).view(-1, 1)
        Reward = torch.tensor(self.ExperienceHistory['reward']).view(-1, 1)
        Done = torch.tensor(self.ExperienceHistory['done']).view(-1, 1)

        
        #加入done

        #計算delta, targetValue, old_predict
        TargetDataset = TensorDataset(State, NextState, Reward, Action, Done)
        TargetLoader = DataLoader(TargetDataset, batch_size=batch_size, shuffle=False)#確保順序一致

        td_target = []
        td_delta = []
        old_predict = []

        self.ValueNetwork.eval()
        self.PolicyNetwork.eval()

        test = True

        for state, nextState, reward, action, done in TargetLoader:
            state = state.to(device, dtype=torch.float32)
            nextState = nextState.to(device, dtype=torch.float32)
            action = action.to(device, dtype=torch.int64)
            reward = reward.to(device, dtype=torch.float32)
            done = done.to(device, dtype=torch.int64)
            
            #計算delta, targetValue 
            V_target = reward + gamma * self.ValueNetwork(nextState).detach() * (1 - done)
            
            d = V_target - self.ValueNetwork(state).detach()

            

            #計算old_predict
            old = torch.log(self.PolicyNetwork(state).gather(1, action)).detach()


            V_target = V_target.to(device="cpu", dtype=torch.float32)
            d = d.to(device="cpu", dtype=torch.float32)
            old = old.to(device="cpu", dtype=torch.float32)

            td_target.extend(list(V_target))
            td_delta.extend(list(d))
            old_predict.extend(list(old))


        #計算advantage

        advantageList = []
        ad = 0
        for d in td_delta[::-1]:
            ad = gamma * lmbda * ad + d
            advantageList.append(ad)

        advantageList.reverse()

        dataset = TensorDataset(State, NextState, Action, Reward, torch.tensor(td_target).view(-1, 1),
                                 torch.tensor(advantageList).view(-1, 1), torch.tensor(old_predict).view(-1, 1))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 訓練模式
        self.ValueNetwork.train()
        self.PolicyNetwork.train()

        totalPolicyLoss = 0
        totalValueLoss = 0

        for _ in range(epochs):
            for state, nextState, action, reward, V_target, advantage, old in dataloader:
                state = state.to(device, dtype=torch.float32)
                state = state.to(device, dtype=torch.float32)
                action = action.to(device, dtype=torch.int64)  # 確保行動是長整型
                reward = reward.to(device, dtype=torch.float32)  # 確保獎勵是浮點數
                V_target = V_target.to(device, dtype=torch.float32)
                advantage = advantage.to(device, dtype=torch.float32)
                old = old.to(device, dtype=torch.float32)
                
                predict = torch.log(self.PolicyNetwork(state).gather(1, action))


                ratio = torch.exp(predict - old)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - episode, 1 + episode) * advantage

                policy_loss = torch.mean(-torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(self.ValueNetwork(state), V_target.detach()))

                optimizer_policy.zero_grad()
                optimizer_value.zero_grad()
                policy_loss.backward()
                value_loss.backward()
                optimizer_policy.step()
                optimizer_value.step()

                totalPolicyLoss += policy_loss.item()
                totalValueLoss += value_loss.item()

        self.PolicyNetwork.to(device="cpu", dtype=torch.float32)
        self.ValueNetwork.to(device="cpu", dtype=torch.float32)
        self.ExperienceHistory['oldstate'] = []
        self.ExperienceHistory['state'] = []
        self.ExperienceHistory['action'] = []
        self.ExperienceHistory['reward'] = []
        self.ExperienceHistory['done'] = []






