import pandas as pd
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random

dfOrig = pd.read_excel('Data/FinalData.xlsx')
scaler = MinMaxScaler()
df = dfOrig.copy()
df[['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Zoom','Focus','Contrast']] = scaler.fit_transform(df[['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Zoom','Focus','Contrast']])


v= {"ol": [0.5*min(df['Orientation_Loss']),1.5*max(df['Orientation_Loss'])],
        "ec": [0.5*min(df['Edge_Coverage']),1.5*max(df['Edge_Coverage'])],
        "at": [0.5*min(df['Average_Thickness']),1.5*max(df['Average_Thickness'])],
        "as": [0.5*min(df['Average_Separation']),1.5*max(df['Average_Separation'])],
        "de": [0.5*min(df['Distance_Entropy']),1.5*max(df['Distance_Entropy'])],
        "z": [0.5*min(df['Zoom']),1.5*max(df['Zoom'])],
        "f": [0.5*min(df['Focus']),1.5*max(df['Focus'])],
        "c": [0.5*min(df['Contrast']),1.5*max(df['Contrast'])]
    }

corr_vals = [list(df.corr()['Zoom'])[:5],list(df.corr()['Focus'])[:5],list(df.corr()['Contrast'])[:5]]

class NewEnv(gym.Env):
    def __init__(self,df,corr_vals):
        self.df = df
        self.corr_vals = corr_vals
        self.agents = ['zoom_agent','focus_agent','contrast_agent']
        self.n_agents = len(self.agents)

        self.state = np.array([self.df[i].to_numpy() for i in ['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Distance_Entropy']])
        #self.state = [np.array([random.uniform(0,1) for _ in range(len(df))])]*5
        #Applied Min-Max scaler so all actions and observations are in range [0,1]
        
        self.action_space = [spaces.Box(low=0.0,high=1.0,dtype=np.float32)]*self.n_agents
        #Values of the 5 target parameters which are continuous
        self.observation_space = [spaces.Box(low=0.0,high=1.0,shape=(len(self.df),),dtype=np.float32)]*5

        self.index = np.random.randint(len(df))

        #Correlations:
        self.gamma_decay = 0.95
        self.gamma = 1.05
        self.low_corr = 0.8
        self.high_corr = 1.2
        self.zoomdeps = {'Orientation_Loss':-0.60,'Edge_Coverage':-0.85,'Average_Thickness':.88,'Average_Separation':0.75}
        self.focusdeps = {'Edge_Coverage':0.44,'Average_Separation':-0.52}
        self.contrastdeps = {'Orientation_Loss': -0.3,'Average_Thickness':0.2}
        # self.zoomRange, self.focusRange, self.contrastRange = 0.1, 0.05, 0.02
    
    def correlations(self,actions,i,factor,param):
        # i: ith action [0,1,2]
        # factor: zoom = 0.1, focus = 0.05, contrast = 0.02
        # param: {"OL": 0, "EC":1, "AT":2, "AS": 3, "DE": 4}
        
        return np.correlate(np.array([random.uniform((1-factor)*actions[i],(1+factor)*actions[i]) for _ in range(len(self.df))]).ravel(),self.state[param].ravel())

    def step(self,actions):
        rewardZ,rewardF,rewardC = 0,0,0
        flagZ,flagF,flagC = [],[],[]


        #Agent- 1
        #OL
        print(self.correlations(actions,0,0.1,0))
        if self.gamma*self.zoomdeps['Orientation_Loss'] <= self.correlations(actions,0,0.1,0).any() <= self.gamma_decay*self.zoomdeps['Orientation_Loss']:
            rewardZ+= 3
            flagZ.append('a')
        
        #EC
        if self.gamma*self.zoomdeps['Edge_Coverage'] <= self.correlations(actions,0,0.1,1).any() <= self.gamma_decay*self.zoomdeps['Edge_Coverage']:
            rewardZ+= 3
            flagZ.append('b')
        
        #AT
        if self.gamma_decay*self.zoomdeps['Average_Thickness'] <= self.correlations(actions,0,0.1,2).any() <= self.gamma*self.zoomdeps['Average_Thickness']:
            rewardZ+= 3
            flagZ.append('c')
            
        #AS
        if self.gamma_decay*self.zoomdeps['Average_Separation'] <= self.correlations(actions,0,0.1,3).any() <= self.gamma*self.zoomdeps['Average_Separation']:
            rewardZ+= 3
            flagZ.append('d')
        
        if flagZ == list(set(['a','b','c','d'])):
            doneZ = True
            flagZ = []
        else:
            doneZ = False
        

        #Agent-2
        #EC
        if self.gamma_decay*self.focusdeps['Edge_Coverage'] <= self.correlations(actions,1,0.05,1).any() <= self.gamma*self.focusdeps['Edge_Coverage']:
            rewardF+= 3
            flagF.append('e')
        
        #AS
        if self.gamma*self.focusdeps['Average_Separation'] <= self.correlations(actions,1,0.05,3).any() <= self.gamma_decay*self.focusdeps['Average_Separation']:
            rewardF+= 3
            flagF.append('f')
        
        if flagF == list(set(['e','f'])):
            doneF = True
            flagF = []
        else:
            doneF = False
        
        #Agent-3
        #OL
        if self.gamma*self.contrastdeps['Orientation_Loss'] <= self.correlations(actions,2,0.02,0).any() <= self.gamma_decay*self.contrastdeps['Orientation_Loss']:
            rewardC+= 3
            flagC.append('g')
        #AT
        if self.gamma_decay*self.contrastdeps['Average_Thickness'] <= self.correlations(actions,2,0.02,2).any() <= self.gamma*self.contrastdeps['Average_Thickness']:
            rewardC+= 3
            flagC.append('g') 
        
        if flagC== list(set(['g','f'])):
            doneC= True
            flagC= []
        else:
            doneC= False
        
        #correlations function calculates the correlation between two arrays,so need to update the state based on agent actions 
        #within the step function. Using actions to calculate the correlations. 

        observations = self.state

        return observations, [rewardZ,rewardF,rewardC], [doneZ,doneF,doneC], {}
        


    def reset(self):
        # return [self.df[i].to_numpy() for i in ['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Distance_Entropy']]
        return [np.array([random.uniform(0,1) for _ in range(len(df))])]*5

    def close(self):
        pass


env = NewEnv(df,corr_vals)
# [np.array([random.uniform(0,1) for _ in range(len(df))])]*5
# action= np.array([0.7,0.7,0.7])
# obs_,reward,done,info= env.step(action)
# print(obs_)