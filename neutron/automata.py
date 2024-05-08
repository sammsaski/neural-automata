from typing import List

class Node:
    """A node in the neural automata."""
    def __init__(self, name: str, nn):
        self.name = name
        self.nn = nn

    def infer(self, x):
        return self.nn(x)

    def __str__(self,):
        return f'Node `{self.name}`'

class Transition:
    """For defining the logic of a transition between nodes."""
    pass

class Neutron:
    """A neural automata."""
    def __init__(self, nodes: List, directed=True):
        """The building of every automata will begin by defining the number of states
        desired.
        """
        self.nodes = nodes 
        self.edges = {}

        self.num_nodes = len(self.nodes)
        self.num_edges = len(self.edges)
    
    def move(self,):
        pass

    def add_node(self, new_node):
        pass

    def build_edge(self, node1: Node, node2: Node):
        self.edges[(node1, node2)]

    def check_neutron(self,):
        pass

    def _update_num_nodes(self,):
        self.num_nodes = len(self.nodes) 

    def _update_num_edges(self,):
        self.num_edges = len(self.edges) 


