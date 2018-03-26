import numpy as np
import networkx as nx

# Rows of the transition matrix
rows = []

# Specify each node's connections (i.e. which states they are connected to)
connections = [(1, 2), (3, 4), (5, 6), [7], [8], [9], [10], [], [], [], []]

# Loop through these connections
for i in connections:
    # create an array of zeros and set the states they're connected to to one
    transitions = np.zeros(len(connections))
    if len(i):
        for j in i:
            transitions[j] = 1
    # add this row of transitions to the list of rows
    rows.append(transitions)

# Convert the list of rows to a 2D array
transition_matrix = np.array(rows)

# To quickly visualise, convert to a networkX graph and draw
G = nx.Graph(transition_matrix)
nx.draw_kamada_kawai(G)