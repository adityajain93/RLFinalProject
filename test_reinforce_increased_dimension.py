from typing import Iterable
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from env_project import toy, diabetes
from matplotlib import pyplot as plt


class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here
        self.num_actions = num_actions
        self.state_dims = state_dims
        print("Number of states / actions ", state_dims, num_actions)
        self.net =  PNet(state_dims, num_actions)
        self.optimizer = optim.Adam(self.net.parameters(),lr = alpha)
       
    def __call__(self,s) -> int:
        
        action = None
        try:
            s_torch = torch.from_numpy(s.reshape((self.state_dims,))).float().unsqueeze(0)
            
            prob = self.net(s_torch).data.cpu().numpy()[0]
            # if np.random.rand() < self.epsilon:
            #     action = np.random.randint(self.num_actions)
            # else:
            action = np.random.choice(np.arange(self.num_actions), p = prob)
        except:
            print(self.net.forward_print(s_torch))
            
        finally:
            return action

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # print(s.shape)
        # print("Delta and Gamma_t " , delta, gamma_t)
        
        s_torch = torch.from_numpy(s.reshape((self.state_dims,))).float().unsqueeze(0)
        # print("Computing probab from net ", self.net(s_torch)[0])

        logloss = -torch.log(self.net(s_torch)[0][a])
        self.optimizer.zero_grad()

        # print("Log Loss ",logloss)
        logloss.backward()

        for param in self.net.parameters():
            # print(param.grad)
            param.grad = param.grad*delta * gamma_t
        self.optimizer.step()
        # return None
        # TODO: implement this method




        

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.net =  VNet(state_dims)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(),lr = alpha)
        self.state_dims = state_dims
        # TODO: implement here

    def __call__(self,s) -> float:
        # TODO: implement this method
        # print(s.shape)
        s_torch = torch.from_numpy(s.reshape((self.state_dims,))).float().unsqueeze(0)
        # print(s_torch.size())
        # TODO: implement this method
        v = self.net(s_torch).data.cpu().numpy()[0][0]
        # do you return this as a tensor
        return v
        # raise NotImplementedError()

    def update(self,s,G):
        # TODO: implement this method
        G = torch.FloatTensor(np.array(G, dtype= np.float64)).unsqueeze(0)
        # print("G is ",G)
        self.optimizer.zero_grad()
        s_tau = torch.FloatTensor(s.reshape((self.state_dims,))).unsqueeze(0)
        v_hat = self.net(s_tau)

        # print("V_hat ", v_hat)
        loss =  self.criterion(G, v_hat[-1])
        loss.backward()
        self.optimizer.step()
        return None

        


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # TODO: implement this method
    # print( "Number of episodes ", num_episodes)
    G_0 = []
    for i_episode in range(num_episodes):
        if i_episode % 100 == 0:
            print("in episode ", i_episode)
        st = env.reset()
        complete_flag = False
        reward_sequene = [0]
        state_sequence = [st]
        action_sequence = []
        t = 0
        while not complete_flag:
            
            a_t = pi(st)
            s_t1, r_t1, complete_flag =  env.step(a_t)
            state_sequence.append(s_t1)
            reward_sequene.append(r_t1)
            action_sequence.append(a_t)
            st = s_t1
            t+=1
        T = t

        for t in range(T):
            G = 0.0
            for k in range(t+1, T+1):
                G += pow(gamma, k-t-1)*reward_sequene[k]
            delta = G -V(state_sequence[t])
            V.update(state_sequence[t], G)
            pi.update(state_sequence[t],action_sequence[t], gamma ** t, delta)
            if t == 0:
                G_0.append(G)

    return G_0, pi, V

        # print(state_sequence)








    # raise NotImplementedError()


class PNet(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(PNet, self).__init__()
        self.num_hidden = 32
        self.hidden1 = nn.Linear(state_dim, self.num_hidden)
        self.hidden2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.output = nn.Linear(self.num_hidden, num_actions)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.softmax(self.output(x))
        return x
        
    def forward_print(self, x):
        x1 = F.relu(self.hidden1(x))
        x2= F.relu(self.hidden2(x1))
        x3 = F.softmax(self.output(x2))
        return x3,x2,x1


class VNet(nn.Module):
    def __init__(self, state_dim):
        super(VNet, self).__init__()
        self.num_hidden = 32
        self.hidden1 = nn.Linear(state_dim, self.num_hidden)
        self.hidden2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.output = nn.Linear(self.num_hidden, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x



def test_reinforce(dataset):
    env = toy(dataset)
    gamma = 1.0
    alpha = 3e-4


    pi = PiApproximationWithNN(
        env.state_dimensions,
        env.num_actions,
        alpha)

    
    B = VApproximationWithNN(
            env.state_dimensions,
            alpha)
    

    return REINFORCE(env,gamma,1000,pi,B)




def test_policy(pi, V, plot = False):
    x = np.arange(0.2, 0.51, 0.1)
    y = np.arange(0.2, 0.91, 0.1)
    points = []
    for x_i in x:
        for y_i in y:
            points.append([x_i, y_i])
    points = np.array(points)
    for point in points:
        s_torch = torch.from_numpy(point.reshape((pi.state_dims,))).float().unsqueeze(0)
        # print(s_torch.size())
        prob = pi.net(s_torch).data.cpu().numpy()[0]
        print("point is  ",point,  "prob is ", prob, "Best policy", pi(point.T) , V(point.T))
    
    if plot:
        for point in points:
            plt.plot(point[0], point[1])
            best_action  = pi(point.T)
            
            if (best_action ==0):
                rep = 'left'
            elif (best_action ==1):
                rep = 'right'
            elif (best_action ==2):
                rep = 'down'
            else:
                rep = 'up'
            plt.text(point[0], point[1], rep, horizontalalignment='center',verticalalignment='center')
            plt.xlim((0,1))
            plt.ylim((0,1))

        plt.show()
    

if __name__ == "__main__":
    num_iter = 5
    all_datasets_training = []
    fig,ax = plt.subplots()
    # plt.ylim(90,100)
    ax.set_xlabel('iteration')
    ax.set_ylabel('Average return ')
    for dim in range(2,8):
        dataset = "data/toy_linear_" + str(dim) + ".csv"
        with_baseline = []
        for _ in range(num_iter):
            
            training_progress, pi, V = test_reinforce(dataset)
            # print(training_progress)
            with_baseline.append(training_progress)
            
        with_baseline = (np.mean(with_baseline,axis=0)).tolist()
        ax.plot(np.arange(len(with_baseline)),with_baseline, label=("Input dimensions "+str(dim )))
        all_datasets_training.append(with_baseline)
    
   
    ax.legend()

    plt.show()

    