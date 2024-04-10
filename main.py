import numpy as np
import pandas as pd
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from newEnv import NewEnv
import matplotlib.pyplot as plt

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state
df = pd.read_excel('Data/FinalData.xlsx')

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
    n_actions = 3
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 50
    N_GAMES = 5000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0
    rewards_list_1, rewards_list_2, rewards_list_3 = [], [], []

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        print(f'Game number: {i}')
        obs = env.reset()
        score = 0
        done = [False]*n_agents
        episode_step = 0
        for _ in range(500):
            if evaluate:
                env.render()
            actions = maddpg_agents.choose_action(obs)
            
            obs_, reward, done, info = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)
            # print('----------------------')
            # print(f'state: {state.shape}')
            # print('----------------------')
            # print(f'state_: {state_.shape}')
            # print('----------------------')
            reward_1 = reward[0]
            reward_2 = reward[1]
            reward_3 = reward[2]

            if episode_step >= MAX_STEPS:
                done = [True]*n_agents

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        rewards_list_1.append(reward_1)
        rewards_list_2.append(reward_2)
        rewards_list_3.append(reward_3)
        avg_score = np.mean(score_history[-100:])
        
        if not evaluate:
            
            if avg_score > best_score:
                
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        
        
        if i % PRINT_INTERVAL == 0 and i > 0:
            
            print('episode', i, 'average score {:.1f}'.format(avg_score))
            # print(1)

    print(f'Total Score: {score_history}')
    plt.plot(rewards_list_1, label='Agent 1')
    plt.plot(rewards_list_2, label='Agent 2')
    plt.plot(rewards_list_3, label='Agent 3')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards for Each Agent')
    plt.legend()
    plt.savefig('rewards_plot.png') 
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