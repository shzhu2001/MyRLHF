import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class DPOAgent:
    def __init__(self,policy,preference_model):
        self.policy = policy
        self.preference = preference_model
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.preference_optimizer = optim.Adam(self.preference.parameters(), lr=1e-3)
        self.gamma = 0.99
    
    def get_action(self,state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        action = torch.distributions.Categorical(probs).sample()
        return action.item()
    
    def train(self,env,episodes):
        for episode in range(episodes):
            state = env.reset()
            done = False
            rewards = []
            actions = []
            episodes_rewards = []
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)
                actions.append(action)
                episodes_rewards.append(reward)
                preference = self.preference(torch.tensor([reward],dtype=torch.float32))
                preference_loss = F.mse_loss(preference,torch.tensor([reward],dtype=torch.float32))
                
                self.preference_optimizer.zero_grad()
                preference_loss.backward()
                self.preference_optimizer.step()    
                
                state = next_state

                if done:
                    total_reward = torch.sum(torch.tensor(episodes_rewards))
                    policy_loss = -torch.mean(torch.log(self.policy(torch.tensor(state))[0][action])*total_reward)
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()
                if episode % 100 == 0:
                    print(f"Episode {episode} Policy Loss: {policy_loss} Preference Loss: {preference_loss}")   
                
                    