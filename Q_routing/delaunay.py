from scipy.spatial import Delaunay
import numpy as np

class delaunay:
    
    def __init__(self, seed=None, size=None, flag=None):
        random = np.random.RandomState(seed)
        rnd = random.uniform(0, 100, size=(size, 2))
        self.points = np.round(rnd, 2)
        self.tri = Delaunay(self.points)
        self.flag = flag
        
    def compute_adj_eucl_dist(self):
       adj_matrix = np.zeros((self.points.shape[0], self.points.shape[0]))
       neighbor_eucl_dist = np.zeros((self.points.shape[0], self.points.shape[0])) 
       for vertex in self.tri.simplices:
           for i in range(3):
               edge_idx1 =  vertex[i]
               edge_idx2 =  vertex[(i+1)%3]
               dist = np.round(np.linalg.norm(self.points[edge_idx1] - self.points[edge_idx2]), 2)
               adj_matrix[edge_idx1, edge_idx2] = 1
               adj_matrix[edge_idx2, edge_idx1] = 1  
               neighbor_eucl_dist[edge_idx1, edge_idx2] = dist
               neighbor_eucl_dist[edge_idx2, edge_idx1] = dist  
           
       return adj_matrix, neighbor_eucl_dist
    
    # Reorder nodes along x coordinates
    def node_ordering(self):
        coords = self.points[self.tri.simplices]
        self.new_points = []
        for coord in coords:
            new_coord = np.zeros_like(coord)
            len = coord.shape[0]
            i = 0
            while len !=0:
                idx = np.argmin(coord[0:3-i,0])
                new_coord[i,:] = coord[idx,:]
                coord = np.delete(coord, idx, 0)
                i+=1
                len = coord.shape[0]
            self.new_points.append(new_coord)
            
        return self.new_points
    
    def compute_edge_info(self, origin, target):
        # Compute dictionary for Dirkjstra and A_Star path search
        if self.flag == None:
            vertex_edges = {}
            for points in self.new_points:
                for i in range(3):
                    point1 = points[i,:]
                    point2 = points[(i+1)%3,:]
            
                    if (point1[0],point1[1]) == origin:
                        dist = np.round(np.linalg.norm(point1 - point2), 2)
                        if (point1[0],point1[1]) in vertex_edges.keys():
                            if ([(point2[0], point2[1]),{'distance': dist}]) in vertex_edges[(point1[0],point1[1])]:
                                continue
                            else:
                                vertex_edges[(point1[0], point1[1])].append([(point2[0], point2[1]),{'distance': dist}])
                        else:
                            vertex_edges[(point1[0], point1[1])] = [[(point2[0], point2[1]),{'distance': dist}]]
                    
                    if (point2[0],point2[1]) == origin:
                        dist = np.round(np.linalg.norm(point1 - point2), 2)
                        if (point2[0],point2[1]) in vertex_edges.keys():
                            if ([(point1[0], point1[1]),{'distance': dist}]) in vertex_edges[(point2[0],point2[1])]:
                                continue
                            else:
                                vertex_edges[(point2[0], point2[1])].append([(point1[0], point1[1]),{'distance': dist}])
                        else:
                            vertex_edges[(point2[0], point2[1])] = [[(point1[0], point1[1]),{'distance': dist}]]
                    
                    if (point1[0],point1[1]) != origin and (point2[0], point2[1]) == target:
                        dist = np.round(np.linalg.norm(point1 - point2), 2) 
                        if (point1[0],point1[1]) in vertex_edges.keys():
                            if ([(point2[0], point2[1]),{'distance': dist}]) in vertex_edges[(point1[0],point1[1])]:
                                continue
                            else: 
                                vertex_edges[(point1[0], point1[1])].append([(point2[0], point2[1]),{'distance': dist}])
                        else:
                            vertex_edges[(point1[0], point1[1])] = [[(point2[0], point2[1]),{'distance': dist}]]
            
                    if (point1[0],point1[1]) == target and (point2[0], point2[1]) != origin:
                        dist = np.round(np.linalg.norm(point1 - point2), 2) 
                        if (point2[0],point2[1]) in vertex_edges.keys():
                            if ([(point1[0], point1[1]),{'distance': dist}]) in vertex_edges[(point2[0],point2[1])]:
                                continue
                            else: 
                                vertex_edges[(point2[0], point2[1])].append([(point1[0], point1[1]),{'distance': dist}])
                        else:
                            vertex_edges[(point2[0], point2[1])] = [[(point1[0], point1[1]),{'distance': dist}]]
                    
                    if (point1[0],point1[1]) != origin and (point1[0],point1[1]) != target:
                        dist = np.round(np.linalg.norm(point1 - point2), 2) 
                        if (point1[0],point1[1]) in vertex_edges.keys():
                            if ([(point2[0], point2[1]),{'distance': dist}]) in vertex_edges[(point1[0],point1[1])]:
                                continue
                            else: 
                                vertex_edges[(point1[0], point1[1])].append([(point2[0], point2[1]),{'distance': dist}])
                        else:
                            vertex_edges[(point1[0], point1[1])] = [[(point2[0], point2[1]),{'distance': dist}]]
                    
                    if (point2[0],point2[1]) != origin and (point2[0],point2[1]) != target:
                        dist = np.round(np.linalg.norm(point1 - point2), 2) 
                        if (point2[0],point2[1]) in vertex_edges.keys():
                            if ([(point1[0], point1[1]),{'distance': dist}]) in vertex_edges[(point2[0],point2[1])]:
                                continue
                            else:
                                vertex_edges[(point2[0], point2[1])].append([(point1[0], point1[1]),{'distance': dist}])
                        else:
                            vertex_edges[(point2[0], point2[1])] = [[(point1[0], point1[1]),{'distance': dist}]] 
        
        # Compute vertices and edges information for agent learning                     
        if self.flag == 1:
            vertex_edges = {}
            for points in self.new_points:
                for i in range(3):
                    point1 = points[i,:]
                    point2 = points[(i+1)%3,:]
            
                    if (point1[0],point1[1]) == origin:
                        value = 0
                        if (point1[0],point1[1]) in vertex_edges.keys():
                            if ([(point2[0], point2[1]),{'value': value}]) in vertex_edges[(point1[0],point1[1])]:
                                continue
                            else:
                                vertex_edges[(point1[0], point1[1])].append([(point2[0], point2[1]),{'value': value}])
                        else:
                            vertex_edges[(point1[0], point1[1])] = [[(point2[0], point2[1]),{'value': value}]]
                    
                    if (point2[0],point2[1]) == origin:
                        value = 0
                        if (point2[0],point2[1]) in vertex_edges.keys():
                            if ([(point1[0], point1[1]),{'value': value}]) in vertex_edges[(point2[0],point2[1])]:
                                continue
                            else:
                                vertex_edges[(point2[0], point2[1])].append([(point1[0], point1[1]),{'value': value}])
                        else:
                            vertex_edges[(point2[0], point2[1])] = [[(point1[0], point1[1]),{'value': value}]]
                    
                    if (point1[0],point1[1]) != origin and (point2[0], point2[1]) == target:
                        value = 0 
                        if (point1[0],point1[1]) in vertex_edges.keys():
                            if ([(point2[0], point2[1]),{'value': value}]) in vertex_edges[(point1[0],point1[1])]:
                                continue
                            else: 
                                vertex_edges[(point1[0], point1[1])].append([(point2[0], point2[1]),{'value': value}])
                        else:
                            vertex_edges[(point1[0], point1[1])] = [[(point2[0], point2[1]),{'value': value}]]
            
                    if (point1[0],point1[1]) == target and (point2[0], point2[1]) != origin:
                        value = 0 
                        if (point2[0],point2[1]) in vertex_edges.keys():
                            if ([(point1[0], point1[1]),{'value': value}]) in vertex_edges[(point2[0],point2[1])]:
                                continue
                            else: 
                                vertex_edges[(point2[0], point2[1])].append([(point1[0], point1[1]),{'value': value}])
                        else:
                            vertex_edges[(point2[0], point2[1])] = [[(point1[0], point1[1]),{'value': value}]]
                    
                    if (point1[0],point1[1]) != origin and (point1[0],point1[1]) != target:
                        value = 0 
                        if (point1[0],point1[1]) in vertex_edges.keys():
                            if ([(point2[0], point2[1]),{'value': value}]) in vertex_edges[(point1[0],point1[1])]:
                                continue
                            else: 
                                vertex_edges[(point1[0], point1[1])].append([(point2[0], point2[1]),{'value': value}])
                        else:
                            vertex_edges[(point1[0], point1[1])] = [[(point2[0], point2[1]),{'value': value}]]
                    
                    if (point2[0],point2[1]) != origin and (point2[0],point2[1]) != target:
                        value = 0 
                        if (point2[0],point2[1]) in vertex_edges.keys():
                            if ([(point1[0], point1[1]),{'value': value}]) in vertex_edges[(point2[0],point2[1])]:
                                continue
                            else:
                                vertex_edges[(point2[0], point2[1])].append([(point1[0], point1[1]),{'value': value}])
                        else:
                            vertex_edges[(point2[0], point2[1])] = [[(point1[0], point1[1]),{'value': value}]]     
                
        return vertex_edges
    
    # Convert vertices and edge information to a state dictionary for learning
    def vertex_edges_to_dict(self,vertex_edges):
        state_dict = {}
        for key in vertex_edges.keys():
            new_dict = {}
            for value in vertex_edges[(key[0], key[1])]:
                new_dict[value[0]] = value[1]['value']
            state_dict[(key)] = new_dict
            del new_dict
        
        return state_dict
    
        

#origin = (0.01, 30.23) # (1.990e+00,2.620e+00)
#goal = (41.7, 72.03) #(90.78, 93.2) #(90.34, 13.75) #(98.89, 74.82)  #(39.68, 53.88)  #(20.45, 87.81)   
#graph = delaunay(seed=1, size=10000)
#graph.node_ordering()
#vertex_edges = graph.compute_edge_info(origin, goal)
#state_dictionary = graph.vertex_edges_to_dict(vertex_edges)
#print(state_dictionary)