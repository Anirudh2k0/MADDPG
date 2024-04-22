import pandas as pd
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

random.seed(42)
dfOrig = pd.read_excel('Data/FinalData.xlsx')
scaler = StandardScaler()
df = dfOrig.copy()
df[['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Zoom','Focus','Contrast']] = scaler.fit_transform(df[['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Zoom','Focus','Contrast']])

print(df.sample())
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
    #  INSTEAD OF ACTIONS AS VALUES OF ZOOM, FOCUS AND CONTRAST, THE ACTIONS ARE CORRELATIONS
    #  WITH THESE CORRELATIONS AND THE DATAFRAME, CONSTRUCT THE ZOOM, FOCUS AND CONTRAST
    def __init__(self,df,corr_vals):

        random.seed(42)
        np.random.seed(42)
        self.df = df
        self.corr_vals = corr_vals
        self.agents = ['zoom_agent','focus_agent','contrast_agent']
        self.n_agents = len(self.agents)

        # self.state = np.array([self.df[i].to_numpy() for i in ['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Distance_Entropy']])
        self.state = np.array([self.df[i].sample().to_numpy() for i in ['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Distance_Entropy']])
        # print('self.state', np.array([self.df[i].to_numpy() for i in ['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Distance_Entropy']]))
        #self.state = [np.array([random.uniform(0,1) for _ in range(len(df))])]*5
        #Applied Min-Max scaler so all actions and observations are in range [0,1]
        
        # self.action_space = [spaces.Box(low=0.0,high=1.0,dtype=np.float32)]*self.n_agents
        self.action_space =  [spaces.Box(low=-1.0,high=1.0,dtype=np.float32,shape=(4,))]
        #Values of the 5 target parameters which are continuous
        # self.observation_space = [spaces.Box(low=0.0,high=1.0,shape=(len(self.df),),dtype=np.float32)]*5
        self.observation_space = [spaces.Box(low=0.0,high=1.0,shape=(1,),dtype=np.float32)]*5

        self.index = np.random.randint(len(df))

        #Correlations:
        self.gamma_decay = 0.8
        self.gamma = 1.2
        self.gammaRanges = {
            'zol': [1.1,0.9],
            'zec': [1.2,0.8],
            'zat': [0.8,1.2],
            'zas': [0.8,1.2],
            'fec': [0.7,1.3],
            'fas': [1.3,0.7],
            'col': [1.7,0.7],
            'cat': [0.8,1.2]
        }
        self.low_corr = 0.8
        self.high_corr = 1.2
        self.zoomdeps = {'Orientation_Loss':-0.60,'Edge_Coverage':-0.85,'Average_Thickness':.88,'Average_Separation':0.75}
        self.focusdeps = {'Edge_Coverage':0.44,'Average_Separation':-0.52}
        self.contrastdeps = {'Orientation_Loss': -0.3,'Average_Thickness':0.2}
        # self.zoomRange, self.focusRange, self.contrastRange = 0.1, 0.05, 0.02
    
    def correlations(self,actions,i,factor,param):

        #  THE ACTIONS ARE NOW CORRELATIONS INSTEAD OF ACTUAL VALUES SO DONT USE THIS


        # i: ith action [0,1,2]
        # factor: zoom = 0.2, focus = 0.1, contrast = 0.1
        # param: {"OL": 0, "EC":1, "AT":2, "AS": 3, "DE": 4}
        
        return np.corrcoef(np.array(np.random.uniform((1-factor)*actions[i],(1+factor)*actions[i],len(self.df))).ravel(),self.state[param].ravel())[0][1]

    def step(self,actions):
        
        rewardZ,rewardF,rewardC = 0,0,0
        flagZ,flagF,flagC = [],[],[]
        info = {"zoom": [-100]*4,"focus": [-100]*2,"contrast": [-100]*2}

        # print("actions: ",actions)
        #Agent- 1
        #OL
        #print(self.correlations(actions,0,0.1,0))
  

        #For the choosen actions, 
        # print(self.gammaRanges['zol'][0]*self.zoomdeps['Orientation_Loss'])
        # print(self.correlations(actions,0,0.3,0))
        # print(any(self.gammaRanges['zol'][0]*self.zoomdeps['Orientation_Loss'] <= x <= self.gammaRanges['zol'][1]*self.zoomdeps['Orientation_Loss'] for x in actions[0]))

        # if self.gammaRanges['zol'][0]*self.zoomdeps['Orientation_Loss'] <= actions[0][0] <= self.gammaRanges['zol'][1]*self.zoomdeps['Orientation_Loss']:
        if any(self.gammaRanges['zol'][0]*self.zoomdeps['Orientation_Loss'] <= x <= self.gammaRanges['zol'][1]*self.zoomdeps['Orientation_Loss'] for x in actions[0]):

            # rewardZ+= abs(actions[0][0])
            rewardZ+=abs(self.zoomdeps['Orientation_Loss'])
            # rewardZ+=.2
            flagZ.append('a')
            info['zoom'][0] = min(actions[0].tolist()+[info['zoom'][0]]+[info['zoom'][0]], key=lambda x:abs(x-self.zoomdeps['Orientation_Loss']))
            print("AAA")
        
        #EC
        # if self.gammaRanges['zec'][0]*self.zoomdeps['Edge_Coverage'] <= actions[0][1] <= self.gammaRanges['zec'][1]*self.zoomdeps['Edge_Coverage']:
        if any(self.gammaRanges['zec'][0]*self.zoomdeps['Edge_Coverage'] <= x <= self.gammaRanges['zec'][1]*self.zoomdeps['Edge_Coverage'] for x in actions[0]):
            # rewardZ+= abs(actions[0][1])
            rewardZ+=abs(self.zoomdeps['Edge_Coverage'])
            # rewardZ+=.2
            info['zoom'][1] = min(actions[0].tolist()+[info['zoom'][1]], key=lambda x:abs(x-self.zoomdeps['Edge_Coverage']))
            flagZ.append('b')
            print("BBB")
        
        #AT
        # if self.gammaRanges['zat'][0]*self.zoomdeps['Average_Thickness'] <= actions[0][2] <= self.gammaRanges['zat'][1]*self.zoomdeps['Average_Thickness']:
        if any(self.gammaRanges['zat'][0]*self.zoomdeps['Average_Thickness'] <= x <= self.gammaRanges['zat'][1]*self.zoomdeps['Average_Thickness'] for x in actions[0]):
            # rewardZ+= abs(actions[0][2])
            rewardZ+=abs(self.zoomdeps['Average_Thickness'])
            # rewardZ+=.2
            info['zoom'][2] = min(actions[0].tolist()+[info['zoom'][2]], key=lambda x:abs(x-self.zoomdeps['Average_Thickness']))
            flagZ.append('c')
            print("CCC")
        
        #AS
        # if self.gammaRanges['zas'][0]*self.zoomdeps['Average_Separation'] <= actions[0][3] <= self.gammaRanges['zas'][1]*self.zoomdeps['Average_Separation']:
        if any(self.gammaRanges['zas'][0]*self.zoomdeps['Average_Separation'] <= x <= self.gammaRanges['zas'][1]*self.zoomdeps['Average_Separation'] for x in actions[0]):
            # rewardZ+= abs(actions[0][3])
            rewardZ+=abs(self.zoomdeps['Average_Separation'])
            # rewardZ+=.2
            info['zoom'][3] = min(actions[0].tolist()+[info['zoom'][3]], key=lambda x:abs(x-self.zoomdeps['Average_Separation']))
            flagZ.append('d')
            print("DDD")
        

        if flagZ == list(set(['a','b','c','d'])):
            doneZ = True
            flagZ = []
        else:
            doneZ = False
        

        #Agent-2
        #EC
        # if self.gammaRanges['fec'][0]*self.focusdeps['Edge_Coverage'] <= actions[1][0] <= self.gammaRanges['fec'][1]*self.focusdeps['Edge_Coverage']:
        if any(self.gammaRanges['fec'][0]*self.focusdeps['Edge_Coverage'] <= x <= self.gammaRanges['fec'][1]*self.focusdeps['Edge_Coverage'] for x in actions[1]):
            # rewardF+= abs(actions[1][0])*1.5
            rewardF+= abs(self.focusdeps['Edge_Coverage'])*1.5
            # rewardF+=.3
            info['focus'][0] = min(actions[1].tolist()[:2]+[info['focus'][0]], key=lambda x:abs(x-self.focusdeps['Edge_Coverage']))
            flagF.append('e')
            print(5)
        
        #AS
        # if self.gammaRanges['fas'][0]*self.focusdeps['Average_Separation'] <= actions[1][1] <= self.gammaRanges['fas'][1]*self.focusdeps['Average_Separation']:
        if any(self.gammaRanges['fas'][0]*self.focusdeps['Average_Separation'] <= x <= self.gammaRanges['fas'][1]*self.focusdeps['Average_Separation'] for x in actions[1]):
            # rewardF+= abs(actions[1][1])*1.5
            rewardF+= abs(self.focusdeps['Average_Separation'])*1.5
            # rewardF+=.3
            info['focus'][1] = min(actions[1].tolist()[:2]+[info['focus'][1]], key=lambda x:abs(x-self.focusdeps['Average_Separation']))
            flagF.append('f')
            print(6)
        
        if flagF == list(set(['e','f'])):
            doneF = True
            flagF = []
        else:
            doneF = False
        
        #Agent-3
        #OL
        # if self.gammaRanges['col'][0]*self.contrastdeps['Orientation_Loss'] <= actions[2][0] <= self.gammaRanges['col'][1]*self.contrastdeps['Orientation_Loss']:
        if any(self.gammaRanges['col'][0]*self.contrastdeps['Orientation_Loss'] <= x <= self.gammaRanges['col'][1]*self.contrastdeps['Orientation_Loss'] for x in actions[2]):
            # rewardC+= abs(actions[2][0])*2
            rewardC+= abs(self.contrastdeps['Orientation_Loss'])*3
            # rewardC+=.6
            info['contrast'][0] = min(actions[2].tolist()[:2]+[info['contrast'][0]], key=lambda x:abs(x-self.contrastdeps['Orientation_Loss']))
            flagC.append('g')
            print(7)
        #AT
        # if self.gammaRanges['cat'][0]*self.contrastdeps['Average_Thickness'] <= actions[2][1] <= self.gammaRanges['cat'][1]*self.contrastdeps['Average_Thickness']:
        if any(self.gammaRanges['cat'][0]*self.contrastdeps['Average_Thickness'] <= x <= self.gammaRanges['cat'][1]*self.contrastdeps['Average_Thickness'] for x in actions[2]):
            # rewardC+= abs(actions[2][1])*2
            rewardC+= abs(self.contrastdeps['Average_Thickness'])*3
            # rewardC+=.6
            info['contrast'][1] = min(actions[2].tolist()[:2]+[info['contrast'][1]], key=lambda x:abs(x-self.contrastdeps['Average_Thickness']))
            flagC.append('g') 
            print(8)
        
        if flagC== list(set(['g','f'])):
            doneC= True
            flagC= []
        else:
            doneC= False
        
        #correlations function calculates the correlation between two arrays,so need to update the state based on agent actions 
        #within the step function. Using actions to calculate the correlations. 
            
        # CORRELATIONS BETWEEN VALUES IN THE DF DO NOT CHANGE. SO OBSERVATIONS CAN REMAIN CONSTANT#
        observations = self.state

        return observations, [rewardZ,rewardF,rewardC], [doneZ,doneF,doneC], info
        # print("Hiiiiiii", observations)
        # return observations, [rewardZ,rewardF,rewardC], [False,False,False], info

        


    def reset(self):
        # return [self.df[i].to_numpy() for i in ['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Distance_Entropy']]
        # return [np.array([random.uniform(0,1) for _ in range(len(df))])]*5
        # return [np.random.uniform(0,1,len(df)),np.random.uniform(0,1,len(df)),np.random.uniform(0,1,len(df)),np.random.uniform(0,1,len(df)),np.random.uniform(0,1,len(df))]
        # return np.array([self.df[i].to_numpy() for i in ['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Distance_Entropy']])
       
        # state = np.array([self.df[i].to_numpy() for i in ['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Distance_Entropy']])
        state = np.array([self.df[i].sample().to_numpy() for i in ['Orientation_Loss','Edge_Coverage','Average_Thickness','Average_Separation','Distance_Entropy']])
        # print('self.state:', state)
        return state


    def close(self):
        pass


# env = NewEnv(df,corr_vals)
# [np.array([random.uniform(0,1) for _ in range(len(df))])]*5
# action= np.array([0.7,0.7,0.7])
# obs_,reward,done,info= env.step(action)
# print(obs_)
# print(df.corr())
