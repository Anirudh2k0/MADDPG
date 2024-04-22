import numpy as np
import pandas as pd
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from latest import NewEnv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state
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

if __name__ == '__main__':

    scenario = 'rise_simple_spread'
    
    env = NewEnv(df=df,corr_vals=corr_vals)
    n_agents = env.n_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    # n_actions = env.action_space[0].n
    # n_actions = 3
    n_actions = [4,4,4]   #Correlations [zoom has 4, focus and contrast have 2, the other 2 are just paddings, that the env dont consider]
    # If the n_actions are of diffent shapes, we need to pad them later as tensor does not accept, so just putting 4,4,4 now and neglecting them in environement
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 50
    N_GAMES = 300
    MAX_STEPS = 80
    total_steps = 0
    score_history = []
    reward_1_history,reward_2_history,reward_3_history = [],[],[]
    evaluate = False
    best_score = 0
    

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        print(f'Game number: {i}')
        obs = env.reset()
        score = 0
        reward1,reward2,reward3 = 0,0,0
        done = [False]*n_agents
        episode_step = 0
        # for _ in range(100):
        while not all(done):
        # for _ in range(100):
            if evaluate:
                env.render()
            actions = maddpg_agents.choose_action(obs)
            # print(actions)
            obs_, reward, done, info = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)
            # print('----------------------')
            # print(f'state: {state.shape}')
            # print('----------------------')
            # print(f'state_: {state_.shape}')
            # print('----------------------')
            

            if episode_step >= MAX_STEPS:
                done = [True]*n_agents

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
               
                maddpg_agents.learn(memory)

            obs = obs_
            
            score += sum(reward)
            reward1 += reward[0]
            reward2 += reward[1]
            reward3 += reward[2]
            total_steps += 1
            episode_step += 1

            # print(f'Info: {info}')

        score_history.append(score)
        reward_1_history.append(reward1)
        reward_2_history.append(reward2)
        reward_3_history.append(reward3)
        
        avg_score = np.mean(score_history[-100:])
        
        if not evaluate:
            
            if avg_score > best_score:
                
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        
        
        if i % PRINT_INTERVAL == 0 and i > 0:
            
            print('episode', i, 'average score {:.1f}'.format(avg_score))
            # print(1)
        
        

    print(f'Total Score: {score_history}')
    ##Incremental Score History:  As distance between correlations decreases, reward decreases
    #But that is ok as this is what we want.
    #But to show an incremental value, we need to calculate an incremental score history
    max_score,max_reward1,max_reward2,max_reward3 = max(score_history),max(reward_1_history),max(reward_2_history),max(reward_3_history)
    incremental_score_history = [max_score-i for i in score_history]
    incremental_reward1_history = [max_reward1-i for i in reward_1_history]
    incremental_reward2_history = [max_reward2-i for i in reward_2_history]
    incremental_reward3_history = [max_reward3-i for i in reward_3_history]


    
    plt.figure()
    plt.plot(incremental_reward1_history, label='Agent 1')
    plt.plot(incremental_reward2_history, label='Agent 2')
    plt.plot(incremental_reward3_history, label='Agent 3')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards for Each Agent')
    plt.legend()
    plt.savefig('rewards_plot.png') 

    plt.figure()
    plt.plot(incremental_score_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Score')
    plt.title('Total Score per Episode')
    plt.savefig('total_score_plot.png')

    # plt.show()

# import numpy as np
# from maddpg import MADDPG
# from buffer import MultiAgentReplayBuffer
# from make_env import make_env

# def obs_list_to_state_vector(observation):
#     state = np.array([])
#     for obs in observation:
#         state = np.concatenate([state, obs])
#     return state

# if __name__ == '__main__':
#     #scenario = 'simple'
#     scenario = 'simple_adversary'
#     env = make_env(scenario)
#     n_agents = env.n
#     actor_dims = []
#     for i in range(n_agents):
#         actor_dims.append(env.observation_space[i].shape[0])
#     critic_dims = sum(actor_dims)

#     # action space is a list of arrays, assume each agent has same action space
#     n_actions = env.action_space[0].n
#     maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
#                            fc1=64, fc2=64,  
#                            alpha=0.01, beta=0.01, scenario=scenario,
#                            chkpt_dir='tmp/maddpg/')

#     memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
#                         n_actions, n_agents, batch_size=1024)

#     PRINT_INTERVAL = 500
#     N_GAMES = 50000
#     MAX_STEPS = 25
#     total_steps = 0
#     score_history = []
#     evaluate = False
#     best_score = 0

#     if evaluate:
#         maddpg_agents.load_checkpoint()

#     for i in range(N_GAMES):
#         obs = env.reset()
#         score = 0
#         done = [False]*n_agents
#         episode_step = 0
#         while not any(done):
#             if evaluate:
#                 env.render()
#                 #time.sleep(0.1) # to slow down the action for the video
#             actions = maddpg_agents.choose_action(obs)
#             obs_, reward, done, info = env.step(actions)

#             state = obs_list_to_state_vector(obs)
#             state_ = obs_list_to_state_vector(obs_)

#             if episode_step >= MAX_STEPS:
#                 done = [True]*n_agents

#             memory.store_transition(obs, state, actions, reward, obs_, state_, done)

#             if total_steps % 100 == 0 and not evaluate:
#                 maddpg_agents.learn(memory)

#             obs = obs_

#             score += sum(reward)
#             total_steps += 1
#             episode_step += 1

#         score_history.append(score)
#         avg_score = np.mean(score_history[-100:])
#         if not evaluate:
#             print(111)
#             if avg_score > best_score:
#                 print(222)
#                 maddpg_agents.save_checkpoint()
#                 best_score = avg_score
#         if i % PRINT_INTERVAL == 0 and i > 0:
#             print(333)
#             print('episode', i, 'average score {:.1f}'.format(avg_score))