import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import Delaunay
import numpy as np

seed = 1
random = np.random.RandomState(seed)
rnd = random.uniform(0, 100, size=(100, 2))
points = np.round(rnd, 2)
tri = Delaunay(points)

def path_to_matrix(paths):
    path_to_matrix = np.zeros((len(paths), 2))
    i = 0
    for path in paths:
        path_to_matrix[i,0] = path[0]
        path_to_matrix[i,1] = path[1]
        i+=1 
    x, y = path_to_matrix[:,0], path_to_matrix[:,1]
    return x, y

dijkstra_path = np.load('results/dijkstras_path.npy')
a_star_path = np.load('results/a_star.npy')
agent_path = np.load('results/agent_path.npy')
#greedy_agent_path = np.load('results/greedy_agent_path.npy')
rewards = np.load('results/rewards.npy')

x_dijk, y_dijk = path_to_matrix(dijkstra_path)
x_a_star, y_a_star = path_to_matrix(a_star_path)
x_agent, y_agent = path_to_matrix(agent_path)
#x_greedy, y_greedy = path_to_matrix(greedy_agent_path)
x_origin = 0.01
y_origin = 30.23
x_goal = 98.89
y_goal = 74.82

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,8))

ax[0].triplot(points[:,0], points[:, 1], tri.simplices)
ax[0].plot(points[:, 0], points[:, 1], 'o')
ax[0].plot(x_origin, y_origin, marker='^', markersize=10, color='red')
ax[0].plot(x_goal, y_goal, marker='^', markersize=10, color='red')
ax[0].plot(x_dijk, y_dijk, label='dijkstra', linewidth=2, color='red')
ax[0].plot(x_a_star, y_a_star, '--', label='A_Star', linewidth=2, color='blue')
ax[0].plot(x_agent, y_agent, label='egreedy Agent', linewidth=2, color='orange')
#ax[0].plot(x_greedy, y_greedy, label='greedy Agent', linewidth=2, color='yellow')
ax[0].annotate('start', xy=(0.01, 30.23), xytext=(0.01, 32.00), fontweight='bold')
ax[0].annotate('goal', xy=(98.89, 74.82), xytext=(98.89, 76.00), fontweight='bold')
ax[0].set_title('Delaunay Path Search')
ax[0].set_xlabel('x coordinates')
ax[0].set_ylabel('y coordinates')
ax[0].legend()

ax[1].plot(rewards, label='reward', linewidth=2, color='red')
ax[1].set_title('Agent Accumulated Reward')
ax[1].set_xlabel('Number of runs')
ax[1].set_ylabel('Total average reward per run')
ax[1].grid(True)
ax[1].legend()

plt.savefig('results/path_results.png', dpi=300)

# display animation of paths
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
ax.triplot(points[:,0], points[:, 1], tri.simplices)
ax.plot(points[:, 0], points[:, 1], 'o')
ax.plot(x_origin, y_origin, marker='^', markersize=10, color='red')
ax.plot(x_goal, y_goal, marker='^', markersize=10, color='red')
ax.annotate('start', xy=(0.01, 30.23), xytext=(0.01, 32.00), fontweight='bold')
ax.annotate('goal', xy=(98.89, 74.82), xytext=(98.89, 76.00), fontweight='bold')
ax.set_xlabel('x coordinates')
ax.set_ylabel('y coordinates')
#ax.set_title('E-greedy Agent Path Search')
#ax.set_title('Dijkstra Path Search')
ax.set_title('A Star Path Search')
line, = ax.plot([],[], marker='o', markersize=10, color='blue')

def init():
    line.set_data([], [])
    return line

x, y = path_to_matrix(a_star_path)

def animate(i):
    line.set_data(x[:i], y[:i])
    
    return line

def frames():
    for x_pos, y_pos in zip(x,y):
        yield x_pos, y_pos
        
animation = FuncAnimation(fig, animate, init_func=init, frames=len(x)+1, interval=1000, repeat=True)

# save results
animation.save('results/a_star.gif', writer='imagemagick')
#animation.save('results/dijkstra.gif', writer='imagemagick')
#animation.save('results/egreedy_agent.gif', writer='imagemagick')
#animation.save('results/greedy_agent.gif', writer='imagemagick')

plt.show()




