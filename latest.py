import pandas as pd
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random

random.seed(42)
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
# print("HII ",corr_vals[0][2])
# print([corr_vals[0][:4]]+[[corr_vals[1][1],corr_vals[1][3]]]+[[corr_vals[2][1],corr_vals[1][2]]])

class NewEnv(gym.Env):
    #  INSTEAD OF ACTIONS AS VALUES OF ZOOM, FOCUS AND CONTRAST, THE ACTIONS ARE CORRELATIONS
    #  WITH THESE CORRELATIONS AND THE DATAFRAME, CONSTRUCT THE ZOOM, FOCUS AND CONTRAST
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self,df,corr_vals):

        random.seed(42)
        np.random.seed(42)
        self.df = df
        self.corr_vals = corr_vals
        self.agents = ['zoom_agent','focus_agent','contrast_agent']
        self.n_agents = len(self.agents)

        self.state = [x.copy() for x in corr_vals]
        # self.init_vals = [[-100,-100,-100,-100,-100]]*3

        self.action_space =  [spaces.Box(low=-1.0,high=1.0,dtype=np.float32,shape=(4,))]
        # Values of the 5 target parameters which are continuous
        # self.observation_space = [spaces.Box(low=0.0,high=1.0,shape=(4,),dtype=np.float32)+2*spaces.Box(low=0.0,high=1.0,shape=(2,),dtype=np.float32)]
        self.observation_space = [spaces.Box(low=0.0,high=1.0,shape=(5,),dtype=np.float32)]*3

        self.index = np.random.randint(len(df))

        #Correlations:
        self.gamma_decay = 0.8
        self.gamma = 1.2
        self.gammaRanges = {
            'zol': [1.05,0.95],
            'zec': [1.05,0.95],
            'zat': [0.95,1.05],
            'zas': [0.95,1.05],
            'fec': [0.95,1.05],
            'fas': [1.05,0.95],
            'col': [1.05,0.95],
            'cat': [0.95,1.05]
        }
        self.low_corr = 0.8
        self.high_corr = 1.2
        self.zoomdeps = {'Orientation_Loss':-0.60,'Edge_Coverage':-0.85,'Average_Thickness':.88,'Average_Separation':0.75}
        self.focusdeps = {'Edge_Coverage':0.44,'Average_Separation':-0.52}
        self.contrastdeps = {'Orientation_Loss': -0.3,'Average_Thickness':0.2}
        self.flagZ,self.flagF,self.flagC = [],[],[]
        # self.zoomRange, self.focusRange, self.contrastRange = 0.1, 0.05, 0.02

        #NEW
        self.corr_buffer = {"zoom": [-100]*4,"focus": [-100]*2,"contrast": [-100]*2}
    
    def correlations(self,actions,i,factor,param):

        #  THE ACTIONS ARE NOW CORRELATIONS INSTEAD OF ACTUAL VALUES SO DONT USE THIS

        # i: ith action [0,1,2]
        # factor: zoom = 0.2, focus = 0.1, contrast = 0.1
        # param: {"OL": 0, "EC":1, "AT":2, "AS": 3, "DE": 4}
        
        return np.corrcoef(np.array(np.random.uniform((1-factor)*actions[i],(1+factor)*actions[i],len(self.df))).ravel(),self.state[param].ravel())[0][1]

    def step(self,actions):
        
        rewardZ,rewardF,rewardC = 0,0,0
        
        # info = {"zoom": [-100]*4,"focus": [-100]*2,"contrast": [-100]*2}

        #Agent- 1
        #OL
       
        if any(self.gammaRanges['zol'][0]*self.zoomdeps['Orientation_Loss'] <= x <= self.gammaRanges['zol'][1]*self.zoomdeps['Orientation_Loss'] for x in actions[0]):
        
            self.corr_buffer['zoom'][0] = min(actions[0].tolist()+[self.corr_buffer['zoom'][0]], key=lambda x:abs(x-self.zoomdeps['Orientation_Loss']))
            rewardZ+=abs(self.zoomdeps['Orientation_Loss']-self.corr_buffer['zoom'][0])
            self.flagZ.append('a')
            self.state[0][0] = self.corr_buffer['zoom'][0]

            #####FROM HERE####
            # self.state[0][0] = min(actions[0].tolist()+[self.init_vals[0][0]], key=lambda x:abs(x-self.zoomdeps['Orientation_Loss']))
            # if abs(self.state[0][0]-self.zoomdeps['Orientation_Loss']) < abs(self.init_vals[0][0]-self.zoomdeps['Orientation_Loss']):
            #     self.flagZ.append('a')
            #     self.init_vals[0][0] = self.state[0][0]
            #     rewardZ += abs(self.state[0][0]-self.zoomdeps['Orientation_Loss'])
                
            #     print("AAA")
        
        #EC
        # if self.gammaRanges['zec'][0]*self.zoomdeps['Edge_Coverage'] <= actions[0][1] <= self.gammaRanges['zec'][1]*self.zoomdeps['Edge_Coverage']:
        if any(self.gammaRanges['zec'][0]*self.zoomdeps['Edge_Coverage'] <= x <= self.gammaRanges['zec'][1]*self.zoomdeps['Edge_Coverage'] for x in actions[0]):

            self.corr_buffer['zoom'][1] = min(actions[0].tolist()+[self.corr_buffer['zoom'][1]], key=lambda x:abs(x-self.zoomdeps['Edge_Coverage']))
            rewardZ+=abs(self.zoomdeps['Edge_Coverage']-self.corr_buffer['zoom'][1])
            self.flagZ.append('b')
            self.state[0][1] = self.corr_buffer['zoom'][1]

            #####  FROM HERE  ####
            # self.state[0][1] = min(actions[0].tolist()+[self.init_vals[0][1]], key=lambda x:abs(x-self.zoomdeps['Edge_Coverage']))
            # if abs(self.state[0][1]-self.zoomdeps['Edge_Coverage']) < abs(self.init_vals[0][1]-self.zoomdeps['Edge_Coverage']):
            #     self.init_vals[0][1] = self.state[0][1]
            #     rewardZ+= abs(self.state[0][1]-self.zoomdeps['Edge_Coverage'])
                
            #     self.flagZ.append('b')
            #     print("BBB")
        
        #AT
        # if self.gammaRanges['zat'][0]*self.zoomdeps['Average_Thickness'] <= actions[0][2] <= self.gammaRanges['zat'][1]*self.zoomdeps['Average_Thickness']:
        if any(self.gammaRanges['zat'][0]*self.zoomdeps['Average_Thickness'] <= x <= self.gammaRanges['zat'][1]*self.zoomdeps['Average_Thickness'] for x in actions[0]):
        
            
            self.corr_buffer['zoom'][2] = min(actions[0].tolist()+[self.corr_buffer['zoom'][2]], key=lambda x:abs(x-self.zoomdeps['Average_Thickness']))
            rewardZ+=abs(self.zoomdeps['Average_Thickness']-self.corr_buffer['zoom'][2])
            self.flagZ.append('c')
            self.state[0][2] = self.corr_buffer['zoom'][2]

            #####  FROM HERE  ####
            # self.state[0][2] = min(actions[0].tolist()+[self.init_vals[0][2]], key=lambda x:abs(x-self.zoomdeps['Average_Thickness']))
            # if abs(self.state[0][2]-self.zoomdeps['Average_Thickness']) < abs(self.init_vals[0][2]-self.zoomdeps['Average_Thickness']):
            #     self.init_vals[0][2] = self.state[0][2]
            #     rewardZ+=abs(self.state[0][2]-self.zoomdeps['Average_Thickness'])
                
            #     self.flagZ.append('c')
            #     print("CCC")
        
        #AS
        # if self.gammaRanges['zas'][0]*self.zoomdeps['Average_Separation'] <= actions[0][3] <= self.gammaRanges['zas'][1]*self.zoomdeps['Average_Separation']:
        if any(self.gammaRanges['zas'][0]*self.zoomdeps['Average_Separation'] <= x <= self.gammaRanges['zas'][1]*self.zoomdeps['Average_Separation'] for x in actions[0]):
           
            self.corr_buffer['zoom'][3] = min(actions[0].tolist()+[self.corr_buffer['zoom'][3]], key=lambda x:abs(x-self.zoomdeps['Average_Separation']))
            rewardZ+=abs(self.zoomdeps['Average_Separation']-self.corr_buffer['zoom'][3])
            self.flagZ.append('d')
            self.state[0][3] =  self.corr_buffer['zoom'][3]

            #####  FROM HERE  ####
            # self.state[0][3] = min(actions[0].tolist()+[self.init_vals[0][3]], key=lambda x:abs(x-self.zoomdeps['Average_Separation']))
            # if abs(self.state[0][3]-self.zoomdeps['Average_Separation']) < abs(self.init_vals[0][3]-self.zoomdeps['Average_Separation']):
            #     self.init_vals[0][3] = self.state[0][3]
            #     rewardZ+=abs(self.state[0][3]-self.zoomdeps['Average_Separation'])
                
            #     self.flagZ.append('d')
            #     print("DDD")
        

        if sorted(list(set(self.flagZ))) == ['a','b','c','d']:
            
            doneZ = True
            # rewardZ+=1
            self.flagZ = []
            # self.init_vals[0] = [-100,-100,-100,-100]
            
        else:
            doneZ = False
        

        #Agent-2
        #EC
        # if self.gammaRanges['fec'][0]*self.focusdeps['Edge_Coverage'] <= actions[1][0] <= self.gammaRanges['fec'][1]*self.focusdeps['Edge_Coverage']:
        if any(self.gammaRanges['fec'][0]*self.focusdeps['Edge_Coverage'] <= x <= self.gammaRanges['fec'][1]*self.focusdeps['Edge_Coverage'] for x in actions[1]):
            
            self.corr_buffer['focus'][0] = min(actions[1].tolist()[:2]+[self.corr_buffer['focus'][0]], key=lambda x:abs(x-self.focusdeps['Edge_Coverage']))
            rewardF+=abs(self.focusdeps['Edge_Coverage']-self.corr_buffer['focus'][0])*2
            self.flagF.append('e')
            self.state[1][1] = self.corr_buffer['focus'][0]

            #####  FROM HERE  ####

            # self.state[1][1] = min(actions[1].tolist()[:2]+[self.init_vals[1][1]], key=lambda x:abs(x-self.focusdeps['Edge_Coverage']))
            # if abs(self.state[1][1]-self.focusdeps['Edge_Coverage']) < abs(self.init_vals[1][1]-self.focusdeps['Edge_Coverage']):
            #     self.init_vals[1][1] = self.state[1][1]
            #     self.flagF.append('e')
            #     # rewardF+= abs(self.state[1][1]-self.focusdeps['Edge_Coverage'])
            #     # rewardF+= 1.5
            #     print(5)
        
        #AS
        # if self.gammaRanges['fas'][0]*self.focusdeps['Average_Separation'] <= actions[1][1] <= self.gammaRanges['fas'][1]*self.focusdeps['Average_Separation']:
        if any(self.gammaRanges['fas'][0]*self.focusdeps['Average_Separation'] <= x <= self.gammaRanges['fas'][1]*self.focusdeps['Average_Separation'] for x in actions[1]):
            
            self.corr_buffer['focus'][1] = min(actions[1].tolist()[:2]+[self.corr_buffer['focus'][1]], key=lambda x:abs(x-self.focusdeps['Average_Separation']))
            rewardF+=abs(self.focusdeps['Average_Separation']-self.corr_buffer['focus'][1])*2
            self.flagF.append('f')
            self.state[1][3] = self.corr_buffer['focus'][1]

            #####  FROM HERE  ####

            # self.state[1][3] = min(actions[1].tolist()[:2]+[self.init_vals[1][3]], key=lambda x:abs(x-self.focusdeps['Average_Separation']))
            # if abs(self.state[1][3]-self.focusdeps['Average_Separation']) < abs(self.init_vals[1][3]-self.focusdeps['Average_Separation']):
            #     self.init_vals[1][3] = self.state[1][3]
            #     self.flagF.append('f')
            #     # rewardF+=abs(self.state[1][3])*1.5
            #     # rewardF+= 1.5
            #     print(6)
        
        if sorted(list(set(self.flagF))) == ['e','f']:
            doneF = True
            self.flagF = []
            # self.init_vals[1] = [-100,-100,-100,-100]
            # rewardF+= 1.5
        else:
            doneF = False
        
        #Agent-3
        #OL
        # if self.gammaRanges['col'][0]*self.contrastdeps['Orientation_Loss'] <= actions[2][0] <= self.gammaRanges['col'][1]*self.contrastdeps['Orientation_Loss']:
        if any(self.gammaRanges['col'][0]*self.contrastdeps['Orientation_Loss'] <= x <= self.gammaRanges['col'][1]*self.contrastdeps['Orientation_Loss'] for x in actions[2]):
            # rewardC+= abs(actions[2][0])*2
            # rewardC+= abs(self.contrastdeps['Orientation_Loss'])*3
            # rewardC+=.6
            # info['contrast'][0] = min(actions[2].tolist()[:2]+[info['contrast'][0]], key=lambda x:abs(x-self.contrastdeps['Orientation_Loss']))
            # self.state[2][0] = info['contrast'][0]

            self.corr_buffer['contrast'][0] = min(actions[2].tolist()[:2]+[self.corr_buffer['contrast'][0]], key=lambda x:abs(x-self.contrastdeps['Orientation_Loss']))
            rewardC+=abs(self.contrastdeps['Orientation_Loss']-self.corr_buffer['contrast'][0])*2
            self.flagC.append('g')
            self.state[2][0] = self.corr_buffer['contrast'][0]

            #####  FROM HERE  ####

            # self.state[2][0] = min(actions[2].tolist()[:2]+[self.init_vals[2][0]], key=lambda x:abs(x-self.contrastdeps['Orientation_Loss']))
            # if abs(self.state[2][0]-self.contrastdeps['Orientation_Loss']) < abs(self.init_vals[2][0]-self.contrastdeps['Orientation_Loss']):
            #     self.init_vals[2][0] = self.state[2][0]
            #     self.flagC.append('g')
            #     # rewardC+=abs(self.state[2][0])*2
            #     # rewardC+= 2
            #     print(7)


        #AT
        # if self.gammaRanges['cat'][0]*self.contrastdeps['Average_Thickness'] <= actions[2][1] <= self.gammaRanges['cat'][1]*self.contrastdeps['Average_Thickness']:
        if any(self.gammaRanges['cat'][0]*self.contrastdeps['Average_Thickness'] <= x <= self.gammaRanges['cat'][1]*self.contrastdeps['Average_Thickness'] for x in actions[2]):
            # rewardC+= abs(actions[2][1])*2
            # rewardC+= abs(self.contrastdeps['Average_Thickness'])*3
            # rewardC+=.6
            # info['contrast'][1] = min(actions[2].tolist()[:2]+[info['contrast'][1]], key=lambda x:abs(x-self.contrastdeps['Average_Thickness']))
            # self.state[2][1] = info['contrast'][1]

            self.corr_buffer['contrast'][1] = min(actions[2].tolist()[:2]+[self.corr_buffer['contrast'][1]], key=lambda x:abs(x-self.contrastdeps['Average_Thickness']))
            rewardC+=abs(self.contrastdeps['Average_Thickness']-self.corr_buffer['contrast'][1])*2
            self.flagC.append('h')
            self.state[2][2] = self.corr_buffer['contrast'][1]

            #####  FROM HERE  ####

            # self.state[2][2] = min(actions[2].tolist()[:2]+[self.init_vals[2][2]], key=lambda x:abs(x-self.contrastdeps['Average_Thickness']))
            # if abs(self.state[2][2]-self.contrastdeps['Average_Thickness']) < abs(self.init_vals[2][2]-self.contrastdeps['Average_Thickness']):
            #     self.init_vals[2][2] = self.state[2][2]
            #     # rewardC+=abs(self.state[2][2])*2
            #     # rewardC+= 2
            #     self.flagC.append('h') 
            #     print(8)
        
        if sorted(list(set(self.flagC)))== ['g','h']:
            doneC= True
            # rewardC+= 2
            self.flagC= []
            # self.init_vals[2] =[-100]*4
        else:
            doneC= False
        
        #correlations function calculates the correlation between two arrays,so need to update the state based on agent actions 
        #within the step function. Using actions to calculate the correlations. 
            

        observations = self.state
        # print(observations)
        
        return observations, [rewardZ,rewardF,rewardC], [doneZ,doneF,doneC], {}
        # print("Hiiiiiii", observations)
        # return observations, [rewardZ,rewardF,rewardC], [False,False,False], {}

        


    def reset(self):
        state = self.corr_vals
        # self.init_vals = [[-100,-100,-100,-100,-100]]*3
        # self.corr_buffer = {"zoom": [-100]*4,"focus": [-100]*2,"contrast": [-100]*2}
        return state


    def close(self):
        pass

    def render(self):
        pass

# env = NewEnv(df,corr_vals)  


# import pandas as pd
# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# from sklearn.preprocessing import MinMaxScaler
# import random

# random.seed(42)


# class NewEnv(gym.Env):
#     #  INSTEAD OF ACTIONS AS VALUES OF ZOOM, FOCUS AND CONTRAST, THE ACTIONS ARE CORRELATIONS
#     #  WITH THESE CORRELATIONS AND THE DATAFRAME, CONSTRUCT THE ZOOM, FOCUS AND CONTRAST
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

#     def __init__(self, df, corr_vals):

#         random.seed(42)
#         np.random.seed(42)
#         self.df = df
#         self.corr_vals = corr_vals
#         self.agents = ['zoom_agent', 'focus_agent', 'contrast_agent']
#         self.n_agents = len(self.agents)

#         self.state = [x.copy() for x in corr_vals]
#         self.init_vals = [0] * len(self.df.columns) * self.n_agents  # Initialize with neutral values

#         self.action_space = [spaces.Box(low=-1.0, high=1.0, dtype=np.float32, shape=(4,))]
#         # Observation space: Each agent observes all correlations
#         self.observation_space = [spaces.Box(low=0.0, high=1.0, shape=(len(self.df.columns) * 5,), dtype=np.float32)]*3

#         self.index = np.random.randint(len(self.df))

#         # Correlations:
#         self.gamma_decay = 0.8
#         self.gamma = 1.2
#         self.gammaRanges = {
#             'zol': [1.05, 0.95],
#             'zec': [1.05, 0.95],
#             'zat': [0.95, 1.05],
#             'zas': [0.95, 1.05],
#             'fec': [0.95, 1.05],
#             'fas': [1.05, 0.95],
#             'col': [1.05, 0.95],
#             'cat': [0.95, 1.05]
#         }
#         self.low_corr = 0.8
#         self.high_corr = 1.2
#         self.zoomdeps = {'Orientation_Loss': -0.60, 'Edge_Coverage': -0.85, 'Average_Thickness': .88, 'Average_Separation': 0.75}
#         self.focusdeps = {'Edge_Coverage': 0.44, 'Average_Separation': -0.52}
#         self.contrastdeps = {'Orientation_Loss': -0.3, 'Average_Thickness': 0.2}
#         self.episode_length = 50  # Define a maximum episode length
#         self.step_count = 0

#     def correlations(self, actions, agent_id):
#         """
#         Calculates the correlation between a subset of features based on agent actions.

#         Args:
#             actions (list): The actions taken by the agent.
#             agent_id (int): The ID of the agent.

#         Returns:
#             list: A list containing the calculated correlations for the agent.
#         """

#         correlations = []
#         start_idx = agent_id * len(self.df.columns)
#         end_idx = start_idx + len(self.df.columns)
#         for i in range(len(self.df.columns)):
#             corr = np.corrcoef(np.array(np.random.uniform((1 - 0.1) * actions[i], (1 + 0.1) * actions[i], len(self.df))).ravel(), self.df.iloc[self.index, start_idx + i:end_idx + i].ravel())[0][1]
#             correlations.append(corr)
#         return correlations

#     def step(self, actions):
#         self.step_count += 1

#         rewardZ, rewardF, rewardC = 0, 0, 0

#         # Update state based on agent actions and calculate correlations
#         for agent_id in range(self.n_agents):
#                     # Update state based on agent actions and calculate correlations (continued)
#             correlations = self.correlations(actions[agent_id], agent_id)
#             start_idx = agent_id * len(self.df.columns)
#             end_idx = start_idx + len(self.df.columns)
#             self.state[agent_id][0:len(self.df.columns)] = correlations

#             # Calculate individual agent rewards based on achieved correlations and dependencies
#             for i in range(len(self.df.columns)):
#                 corr = correlations[i]
#                 if agent_id == 0:  # zoom agent
#                     if 'z' in self.df.columns.iloc[start_idx + i]:
#                         weight = self.zoomdeps[self.df.columns.iloc[start_idx + i]]
#                         rewardZ += weight * (max(self.gammaRanges['z' + self.df.columns.iloc[start_idx + i][1:]] - corr, 0) +
#                                             min(corr - self.gammaRanges['z' + self.df.columns.iloc[start_idx + i][1:]], 0)) * self.gamma_decay ** i
#                 elif agent_id == 1:  # focus agent
#                     if 'f' in self.df.columns.iloc[start_idx + i]:
#                         weight = self.focusdeps[self.df.columns.iloc[start_idx + i]]
#                         rewardF += weight * (max(self.gammaRanges['f' + self.df.columns.iloc[start_idx + i][1:]] - corr, 0) +
#                                             min(corr - self.gammaRanges['f' + self.df.columns.iloc[start_idx + i][1:]], 0)) * self.gamma_decay ** i
#                 else:  # contrast agent
#                     if 'c' in self.df.columns.iloc[start_idx + i]:
#                         weight = self.contrastdeps[self.df.columns.iloc[start_idx + i]]
#                         rewardC += weight * (max(self.gammaRanges['c' + self.df.columns.iloc[start_idx + i][1:]] - corr, 0) +
#                                             min(corr - self.gammaRanges['c' + self.df.columns.iloc[start_idx + i][1:]], 0)) * self.gamma_decay ** i

#         # Clip rewards to a reasonable range
#         rewardZ = np.clip(rewardZ, -1.0, 1.0)
#         rewardF = np.clip(rewardF, -1.0, 1.0)
#         rewardC = np.clip(rewardC, -1.0, 1.0)

#         # Update total reward and information dictionary
#         reward = rewardZ + rewardF + rewardC
#         info = {'achieved_correlations': self.state, 'individual_rewards': [rewardZ, rewardF, rewardC]}

#         # Termination criteria: episode length or minimum reward threshold
#         done = self.step_count == self.episode_length or reward < -0.5 * self.n_agents  # Early stopping if reward consistently negative

#         # Update state with achieved correlations for all agents
#         state = np.concatenate([agent_state for agent_state in self.state])

#         return state, reward, done, info

#     def reset(self):
#         self.step_count = 0
#         self.index = np.random.randint(len(self.df))
#         self.state = [x.copy() for x in self.corr_vals]
#         return np.concatenate([agent_state for agent_state in self.state])

#     def render(self, mode="human"):
#         if mode == "human":
#             # Print some human-readable information about the current state
#             print(f"Current Correlations: {self.state}")
#             print(f"Episode Step: {self.step_count}")
#         else:
#             # Render an RGB array for visualization
#             # (implementation omitted for brevity)
#             pass

#     def close(self):
#         pass

