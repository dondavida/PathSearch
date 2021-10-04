from rl_glue import RLGlue
import numpy as np
from tqdm import tqdm as tq
from delaunay import delaunay
from environment import BaseEnvironment
from agent import BaseAgent
from copy import deepcopy
import os

origin = (0.01, 30.23) # (1.990e+00,2.620e+00)
goal = (98.89, 74.82) #(41.7, 72.03) #(90.78, 93.2) #(90.34, 13.75) #(98.89, 74.82)  #(39.68, 53.88)  #(20.45, 87.81)   
graph = delaunay(seed=1, size=100, flag=1)
graph.node_ordering()
vertex_edges = graph.compute_edge_info(origin, goal)
state_dictionary = graph.vertex_edges_to_dict(vertex_edges)
Q_dictionary = deepcopy(state_dictionary)
#print(Q_dictionary)

class DelaunayEnvironment(BaseEnvironment):
    
    def env_init(self, env_info):
        self.start_state = env_info["origin"]
        self.terminal_state = env_info["goal"]
    
    def env_start(self):
        # set self.reward_state_term tuple
        reward = 0.0
        self.current_state = self.start_state
        is_terminal = False
                
        self.reward_state_term = (reward, self.current_state, is_terminal)
        
        # return first state from the environment
        return self.reward_state_term[1]
    
    def env_step(self, action_next_state):
        
        self.last_state = self.reward_state_term[1]
        next_state = action_next_state

        if (next_state[0],next_state[1]) != (self.terminal_state[0],self.terminal_state[1]):
            reward = -10
            is_terminal = False
        
        if (next_state[0],next_state[1]) == (self.terminal_state[0],self.terminal_state[1]):
            reward = 1
            is_terminal = True
        
        self.last_state = next_state  # The current state of the agent 
        self.reward_state_term = (reward, self.last_state, is_terminal)
        
        return self.reward_state_term

def get_step_state_of_max_value(dic, states_visited):
    # create new dictionary excluding the last state 
    new_successor_list = []
    new_dic = {}
    for k,v in dic.items():
        if (k == None):
            continue
        if (k in states_visited):
            continue
        else:
            new_dic[(k[0],k[1])] = v          
    # Randomly choose any visited state, 
    # if the current state has no new states as its values except the visited ones
    if len(new_dic.keys()) == 0:
        for k in dic.keys():
            new_successor_list.append(k)
        num_states = len(new_successor_list)
        state_index = np.random.choice(range(num_states))
        k = new_successor_list[state_index]
        return k
    # choose maximum state value from the new dictionary of non-visited states 
    else:
        max_val = max(new_dic.values())
        key = [k for k, v in new_dic.items() if v == max_val]
        return key[0]

def get_start_state_of_max_value(dic):
    min_val = min(dic.values())
    key = [k for k, v in dic.items() if v == min_val]
    return key[0]
    
def e_greedy_step_action_state(dic, states_visited):
    # create a new list excluding the last state
    new_successor_list = []
    for k in dic.keys():
        if (k in states_visited):
            continue
        else:
            new_successor_list.append(k)     
    # Randomly choose any visited state, 
    # if the current state has no new states as its values except the visited ones       
    if len(new_successor_list) == 0:
        for k in dic.keys():
            new_successor_list.append(k)
        num_states = len(new_successor_list)
        state_index = np.random.choice(range(num_states))
        action_next_state = new_successor_list[state_index]
        return action_next_state
    # Randomly choose any non-visited state
    else:
        num_states = len(new_successor_list)
        state_index = np.random.choice(range(num_states))
        action_next_state = new_successor_list[state_index]
        return action_next_state

def e_greedy_start_action_state(dic):
    # create a new list excluding the last state
    # Randomly choose any non-visited state
    new_successor_list = []
    for k in dic.keys():
        new_successor_list.append(k)
    num_states = len(new_successor_list)
    state_index = np.random.choice(range(num_states))
    action_next_state = new_successor_list[state_index]
    return action_next_state
    
def get_values_of_keys(dic):
    values = []
    for value in dic.values():
        values.append(value)
    return values

def get_keys(dic):
    keys = []
    for key in dic.keys():
        keys.append(key)
    return keys
        
class DelaunayAgent(BaseAgent):
    
    def __init__(self):
        
        self.discount_factor = None
        self.step_size = None
        self.epsilon = None
        self.states_visited = []
    
    def agent_init(self, agent_info):
        
        self.discount_factor = agent_info["discount_factor"]
        self.step_size = agent_info["step_size"]
        self.epsilon = agent_info["epsilon"]
        
    def agent_start(self, start_state):
        
        self.states_visited.append(start_state)
        # choose action using e-greedy
        action_next_state = e_greedy_start_action_state(Q_dictionary[(start_state[0],start_state[1])])    
        self.states_visited.append(action_next_state)
        self.last_state = start_state  # start state of the environment
        self.action_next_state = action_next_state
        
        return action_next_state
            
    def agent_step(self, reward, next_state):
        
        # update weight (cost) of current_state
        current_value = Q_dictionary[(self.last_state[0],self.last_state[1])][(self.action_next_state[0],self.action_next_state[1])]
        delta  = float(reward) + self.discount_factor * max(Q_dictionary[(self.action_next_state[0],self.action_next_state[1])].values()) - current_value
        Q_dictionary[(self.last_state[0],self.last_state[1])][(self.action_next_state[0],self.action_next_state[1])] = current_value + self.step_size * delta 
        
        # choose action using e-greedy
        if np.random.rand() < self.epsilon:
            action_next_state = e_greedy_step_action_state(Q_dictionary[(next_state[0],next_state[1])], self.states_visited)
        else:
            action_next_state = get_step_state_of_max_value(Q_dictionary[(next_state[0],next_state[1])], self.states_visited)
        #action_next_state = get_step_state_of_max_value(Q_dictionary[(next_state[0],next_state[1])], self.states_visited)
        self.states_visited.append(action_next_state)
        self.last_state = next_state
        self.action_next_state = action_next_state
        
        return action_next_state
        
    def agent_end(self, reward):
        
         current_value = Q_dictionary[(self.last_state[0],self.last_state[1])][(self.action_next_state[0], self.action_next_state[1])]
         delta = float(reward) - current_value
         Q_dictionary[(self.last_state[0],self.last_state[1])][(self.action_next_state[0],self.action_next_state[1])] = current_value + self.step_size * delta
         self.states_visited.clear()
         
def run_experiment(environment, agent, experiment_parameters):
        
        rl_glue = RLGlue(environment, agent)
        
        env_info = {
            "origin": experiment_parameters["origin"],
            "goal" : experiment_parameters["goal"]
        }
        
        agent_info = {
            "discount_factor": experiment_parameters["discount_factor"],
            "step_size": experiment_parameters["step_size"],
            "epsilon": experiment_parameters["epsilon"]
        }
        
        all_states_visited = {}
        all_rewards_sum = []
        all_exp_avg_reward = []
        for run in tq(range(1, experiment_parameters["num_runs"]+1)):
            rl_glue.rl_init(agent_info, env_info)
            rewards_sum = 0
            exp_avg_reward_per_episode = []
            states_visited_per_episode = {}
            for episode in range(1, experiment_parameters["num_episodes"]+1):
                states_visited = []
                if episode < experiment_parameters["num_episodes"]-10:
                    # Run an episode
                    rl_glue.rl_episode(0)
                else:
                    # Runs an episode while keeping track of visited states
                    states_visited.append(experiment_parameters['origin'])
                    last_state, action_state = rl_glue.rl_start()
                    states_visited.append(action_state)
                    is_terminal = False
                    while not is_terminal:
                        reward, state, action_state, is_terminal = rl_glue.rl_step()
                        if action_state == None:
                            continue
                        else:
                            states_visited.append(action_state)
                        
                rewards_sum += rl_glue.rl_return()        
                states_visited_per_episode[episode] = states_visited
            all_states_visited[run] = states_visited_per_episode
            
                
            #all_states_visted[run] = states_visited_per_episode
            all_rewards_sum.append(rewards_sum/experiment_parameters["num_episodes"])
            #print(all_rewards_sum)
            
        return all_states_visited, all_rewards_sum
            
def path_to_matrix(paths):
    
    path_to_matrix = np.zeros((len(paths), 2))
    i = 0
    for path in paths:
        path_to_matrix[i,0] = path[0]
        path_to_matrix[i,1] = path[1]
        i+=1
    
    return path_to_matrix

def num_state_occurence(state_episode_list):
    
    state_number_dict = {}
    for state in state_episode_list:
        if state == None:
            continue
        if (state[0],state[1]) in state_number_dict.keys():
            continue
        else:
            count = state_episode_list.count((state[0],state[1]))
            state_number_dict[(state[0],state[1])] = count
    
    return state_number_dict
            
        
experiment_parameters = {
    "origin": (0.01, 30.23),
    "goal": (98.89, 74.82),
    "discount_factor": 0.90,
    "step_size": 0.1,
    "epsilon": 0.1,
    "num_episodes": 20000,
    "num_runs" : 500
}    

states, rewards = run_experiment(DelaunayEnvironment, DelaunayAgent, experiment_parameters)

# save states and rewards
if os.path.exists('results'):
    pass
if not os.path.exists('results'):
    os.makedirs('results')
np.save('results/agent_path.npy', states[experiment_parameters['num_runs']][experiment_parameters['num_episodes']])
np.save('results/greedy_rewards.npy', rewards)


#path_matrix = post_process_path(states[experiment_parameters["num_runs"]][-1])
#x_range = 1000
#data_mean = np.array([np.mean(rewards)])
#print(data_mean)
#data_std_err = np.array([np.std(rewards)/np.sqrt(len(rewards))])
#data_mean = data_mean[:x_range]
#data_std_err = data_std_err[:x_range]
#plt_x_range = range(0,len(data_mean))[:x_range]
#plt.plot(plt_x_range, np.mean(rewards, axis=0))
#print(rewards)
#plt.plot(rewards)
#plt.fill_between(plt_x_range, data_mean - data_std_err, data_mean + data_std_err, alpha = 0.2)
#plt.fill_between(data_mean - data_std_err, data_mean + data_std_err, alpha = 0.2)
#plt.show()
       