from delaunay import delaunay
import numpy as np
import priority_dict
import os

origin = (0.01, 30.23) # (1.990e+00,2.620e+00)
goal =  (98.89, 74.82) #(41.7, 72.03) #(90.78, 93.2) #(90.34, 13.75) #(98.89, 74.82)  #(39.68, 53.88)  #(20.45, 87.81)   
network = delaunay(seed=1, size=100)
network.node_ordering()
vertex_edges = network.compute_edge_info(origin, goal)

class PathSearch:
    def __init__(self, graph):
        self.graph = graph
    
    # ## Dijkstra's Search
    # This function is implemented in `dijkstras_search()`.
    # Helper function `get_path()` assists in retrieving the path
    # from the dictionary of predecessors once the goal is found.
    # To perform Dijkstra's search, a priority queue is required(or a min heap),
    # which is defined as the `priority_dict` class. This class is accessed just as a standard dictionary is,
    # except that it orders the keys by their value. Vertices are used as the keys to priority queue,
    # and their distance from the start as their value. For example, to set the distance of vertex `v` to
    # the variable `dist`, we can do `open_queue[v] = dist`. To get the smallest value in the priority queue,
    # `priority_dict.pop_smallest()` is used.
    # This returns a tuple of the vertex key and it's distance from the origin.
    def dijkstras_search(self, origin, goal):
        
        # The priority queue of open vertices reached.
        # Keys are the vertex keys, vals are the distances.
        open_queue = priority_dict.priority_dict({})
        
        # The dictionary of closed vertices processed.
        closed_dict = {}
        
        # The dictionary of predecessors for each vertex.
        predecessors = {}
        
        # Add the origin to the open queue.
        open_queue[origin] = 0.0

        # Iterate through the open queue, until goal is found.
        # Each time, perform a Dijkstra's update on the queue.
        goal_found = False
        while (open_queue):
            # u = preceeding vertex, ucost = distance of the preceeding vertex to a preceeded vertex
            u, uCost = open_queue.pop_smallest() 
            
            if u == goal:
                goal_found = True
                break
                
            for edge in self.graph[(u)]:
                v = edge[0] # successor vertex
                
                if (v[0],v[1]) in closed_dict:
                    continue
                    
                uvCost = edge[1]['distance'] # distance
                
                if v in open_queue:
                    if uCost + uvCost < open_queue[v]:
                        open_queue[v] = uCost + uvCost
                        predecessors[v] = u
                else:
                    open_queue[v] = uCost + uvCost
                    predecessors[v] = u
                    
            closed_dict[u] = 1
        
        # If goal not found after getting through the entire priority queue,
        # something is wrong.
        if not goal_found:
            raise ValueError("Goal not found in search.")
        
        # Construct the path from the predecessors dictionary.
        return self.get_path(origin, goal, predecessors)  
        
    
    # ## A* Search
    # Distance heuristic is employed to implement A* search for path search problem.
    # For a given graph, origin vertex key, and goal vertex key,
    # computes the shortest path in the graph from the origin vertex
    # to the goal vertex using A* search. 
    # Returns the shortest path as a list of vertex keys.
    def a_star_search(self, origin, goal):
        # The priority queue of open vertices reached.
        # Keys are the vertex keys, vals are the accumulated
        # distances plus the heuristic estimates of the distance
        # to go.
        open_queue = priority_dict.priority_dict({})
        
        # The dictionary of closed vertices we've processed.
        closed_dict = {}
        
        # The dictionary of predecessors for each vertex.
        predecessors = {}
        
        # The dictionary that stores the best cost to reach each
        # vertex found so far.
        costs = {}
        
        # Add the origin to the open queue and the costs dictionary.
        costs[origin] = 0.0
        open_queue[origin] = self.distance_heuristic(origin, goal)

        # Iterate through the open queue, until we find the goal.
        # Each time, perform an A* update on the queue.
        goal_found = False
        while (open_queue):
            u, u_heuristic = open_queue.pop_smallest()
            uCost = costs[u] 
            
            if u == goal:
                goal_found = True
                break
                
            for edge in self.graph[(u)]:
                v = edge[0]
                if v in closed_dict:
                    continue
                    
                uvCost = edge[1]['distance']
                
                if (v[0],v[1]) in open_queue:
                    if uCost + uvCost + self.distance_heuristic(v, goal) < open_queue[v]:
                        open_queue[v] = uCost + uvCost + self.distance_heuristic(v, goal)
                        costs[v] = uCost + uvCost
                        predecessors[v] = u
                else:
                    open_queue[v] = uCost + uvCost + self.distance_heuristic(v, goal)
                    costs[v] = uCost + uvCost
                    predecessors[v] = u
                    
            closed_dict[u] =  1
            
        # If goal not found after getting through the entire priority queue,
        # something is wrong.
        if not goal_found:
            raise ValueError("Goal not found in search.")
        
        # Construct the path from the predecessors dictionary.
        return self.get_path(origin, goal, predecessors)
    
     # Computes the Euclidean distance between two vertices.
    def distance_heuristic(self, state_key, goal_key):
        x1 = state_key[0]
        y1 = state_key[1]
        x2 = goal_key[0]
        y2 = goal_key[1]
        
        d = ((x2-x1)**2 + (y2-y1)**2)**0.5
        
        return d
    
    def get_path(self, origin_key, goal_key, predecessors):
        key = goal_key
        path = [goal_key]
        while (key != origin_key):
            key = predecessors[key]
            path.insert(0, key)
        
        return path  
    
    def path_to_matrix(paths):
        path_to_matrix = np.zeros((len(paths), 2))
        i = 0
        for path in paths:
            path_to_matrix[i,0] = path[0]
            path_to_matrix[i,1] = path[1]
            i+=1
    
        return path_to_matrix
    
path = PathSearch(vertex_edges)
Dijk_path = path.dijkstras_search(origin, goal)
A_Star_path = path.a_star_search(origin, goal)

# save paths
def save_file():
    if os.path.exists('results'):
        pass
    if not os.path.exists('results'):
        os.makedirs('results')
    np.save('results/dijkstras_path.npy', Dijk_path)
    np.save('results/a_star.npy', A_Star_path) 
    
save_file()   


    