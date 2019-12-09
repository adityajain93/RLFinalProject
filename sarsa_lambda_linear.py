# Code for Value Function using Tile Coding adopted from Programming Assignment 3

import numpy as np
import xgboost
from env_project import EnvSpec, Env
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from env_project import diabetes, toy
import matplotlib.pyplot as plt

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.all_tilings = self.create_all_tilings(state_low, state_high, num_tilings, tile_width)
        # print("Tilings created with shape ", self.all_tilings.shape)
        # print(self.all_tilings)
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.state_low = state_low
        self.state_high = state_high
        self.tile_width = tile_width
        self.num_tiles = np.ceil(np.round((state_high-state_low)/tile_width, 2)) + 1
        self.num_tiles = [int(x) for x in self.num_tiles]
        self.tiles_per_tiling = int(np.prod(self.num_tiles))
        self.dim_shape = (self.num_actions,) + (int(num_tilings), ) + tuple(self.num_tiles) 
        # TODO: implement here

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        self.d = self.num_actions * self.num_tilings * self.tiles_per_tiling
        return self.d
    
    def state_vector_len(self) -> int:
        self.num_states = self.num_tilings * self.tiles_per_tiling
        return self.num_states

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """

        if done:
            return np.zeros(self.feature_vector_len())
        else:
            encoding = self.encoded_feature(s,a)
        # print("In Call function ", s, encoding)
            return encoding
        
        # TODO: implement this method
        raise NotImplementedError()

    def one_feature_tiling(self, low, high, tile_width, num_tilings, tiling_index):
        
        num_tiles =  np.ceil(np.round((high-low)/tile_width, 2)) + 1
        low =  (low - tiling_index / num_tilings * tile_width)
        high_new =  low  +   num_tiles * tile_width
        # print(low, high_new, num_tiles)
        return np.linspace(low, high_new, num_tiles, endpoint= False) 

    def create_all_tilings(self, state_low, state_high, num_tilings, tile_width):
        all_tilings =  []
        for tiling_index in range(num_tilings):
            one_tiling = []
            for i in range(len(state_low)):
                # print("In feature with tiling index ", i , tiling_index)
                one_feature_tiling = self.one_feature_tiling(state_low[i], state_high[i], tile_width[i], num_tilings, tiling_index)
                one_tiling.append(one_feature_tiling)
            
            all_tilings.append(one_tiling)
        print("Completed creating all tilings ")
        print(all_tilings)
        return np.array(all_tilings)


    def encoded_feature(self, state, action):
        # print("In encoding feature")
        # encoding = []
        # print(self.dim_shape)
        try:
            s = np.zeros(self.dim_shape, dtype = np.int8)
            index_tuple = [action]
            index_tuple.append(tuple(np.arange(self.num_tilings).astype(int)))
            # print("Index Tuple before feature additions ", index_tuple)
            for i in range(len(state)):
                s_i = state[i]
                low = self.state_low[i]
                width =  self.tile_width[i]
                # print(s_i, low)
                feature = np.floor((s_i - low) / width +  np.arange(self.num_tilings)/self.num_tilings)
                index_tuple.append(tuple(feature.astype(int)))
            # print("Final index ", index_tuple)
            # import time

            # start = time.time()
            s[tuple(index_tuple)] =  1
        # end = time.time()
        # print(end - start)
            s = s.flatten()
        # print(s.shape)
        # print(np.sum(s))
            return s
        except:
            print("Encoding error occured for state ", state, " action ", action)
            print("Final index ", index_tuple)
        finally:
            pass

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,) -> np.array:
    r"""
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.1):
        # nA = env.action_space.n
        nA = env.num_actions
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]
        # print(Q)
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.random.choice(np.flatnonzero(Q == max(Q)))



    w = np.zeros((X.feature_vector_len()))
    w_array = []
    #TODO: implement this function

    # np.random.seed(10)
    for i_episode in range(num_episode):
        if(i_episode % 100 ==0):
            print("-----------------In episode-------------------- ", i_episode)
            w_array.append(w.copy())
        initial_state = env.reset()
        # print("Initial state is ", " , ".join(map(str, initial_state)) ,  "Initial prediction ", env.initial_prediction)
        # Is initial action always false for done flag?
        a_t =  epsilon_greedy_policy(initial_state, False, w, epsilon = 0.20)
        done = False
        x_t = X(initial_state, done, a_t)
        z = np.zeros(X.feature_vector_len())
        Q_old = 0
        t = 0
        s_t1 = None
        
        state_sequence = [initial_state]
        action_sequence = []
        r_t1 = None
        try:
            while not done:
                # print("Step Taken ", t)
                
                s_t1, r_t1, done =  env.step(a_t)
                
                state_sequence.append(s_t1)
                action_sequence.append(a_t)
                # print("reward ", r_t1)
                a_t1 = epsilon_greedy_policy(s_t1, done, w)
                x_t1 = X(s_t1,done, a_t1)
                Q_t = np.dot(w, x_t)
                Q_t1 = np.dot(w, x_t1)
                delta = r_t1 + gamma*Q_t1 - Q_t
                z = gamma * lam * z + (1 - alpha*gamma*lam* (np.dot(z, x_t)) ) * x_t
                w += alpha*(delta + Q_t - Q_old) * z - alpha* (Q_t -Q_old)*x_t
                Q_old = Q_t1
                x_t = x_t1
                a_t = a_t1
                # print("Step Taken ", t)
                t+=1

            # print("T for this episode ", t)
        except:
            print("---------------- Error Occured in Running each step in SARSA Lambda")
            print("Current state is ", env.state, "Action is ", a_t)
            print("next_State is ", s_t1, "reward is ", r_t1)
            # print("state is ", s_t1, "reward is ", r_t1, "action is ", a_t)
            [print(state, action) for state, action in zip(state_sequence,action_sequence)]
            return
        finally:
            pass
    
    return w, np.array(w_array)

def test_algorithm(num_tiles):
    
    env = toy('data/toy_linear.csv')
    
    gamma = 0.98
    tile_width =  (env.state_high - env.state_low)/num_tiles
    X = StateActionFeatureVectorWithTile(
        env.state_low,
        env.state_high,
        env.num_actions,
        num_tilings = 1,
        tile_width=tile_width
    )

    w_final, w_array = SarsaLambda(env, gamma, 0.02, 0.1, X, 10000)
    # print(w_array)
    def greedy_policy(s, w,done):
        Q = [np.dot(w, X(s,done,a)) for a in range(env.num_actions)]
        # print("S is ", s, " Q is  " , Q)
        return np.argmax(Q)
    

    def _eval(w, render=False):
        s, done = env.reset(), False
        if render:
            print("Initial state  ", "_".join(map(str, s)))
        action_sequence = []
        G = 0.
        t = 0 
        while not done:
            if t > 100:
                Q = [np.dot(w, X(s,done,a)) for a in range(env.num_actions)]
                # sprint("Q is ", Q)
                break
            t += 1
            a = greedy_policy(s,w,done)
            action_sequence.append(a)
            s,r,done = env.step(a)
            # if render: env.render()

            G += r
        if render:
            print("Inital prediction is ",env.initial_prediction, "Return is ", G ,  " , ".join(map(str, action_sequence)))
        return G
    perf_average = []
    Gs = [_eval(w_final, True) for _ in  range(100)]
    for i in range(w_array.shape[0]):
        w_i = w_array[i]
        # print(",".join(map(str, w_i)))
        Gs = [_eval(w_i) for _ in  range(50)]
        perf_average.append(np.mean(Gs))
    
    print(perf_average)

    
     
    return perf_average

if __name__ == "__main__":
    average = []
    fig,ax = plt.subplots()
    plt.ylim(90,100)
    ax.set_xlabel('iteration / 100')
    ax.set_ylabel('Average return ')
    np.random.seed(9)
    tiles_list = [2,3, 6, 10]
    for num_tiles in tiles_list:
        np.random.seed(num_tiles)
        results = test_algorithm(num_tiles)
        average.append(results[1:])
        ax.plot(np.arange(len(results)),results, label=str(num_tiles))

    average = np.array(average)
    ax.legend()

    plt.show()
 

    

        
    
