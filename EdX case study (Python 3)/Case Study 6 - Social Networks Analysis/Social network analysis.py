import networkx as nx

#create empty graph
G = nx.Graph()

G.add_node(1)
G.add_nodes_from([2,3]) #add multipul nodes at once
G.add_nodes_from(["u","v"]) #Labels don't have to be numbers
G.nodes() #to check the nodes in a graph G

G.add_edge(1,2)
#add multipul edges at once
#you can add edge to points that don't previously exist
#in that case python will just first create the nodes
G.add_edges_from([(1,3),(1,4),(1,5),(1,6)])
G.add_edge("u","v")
G.add_edge("u","w")
G.edges()

G.remove_node(2)
G.remove_nodes_from([4,5])

G.remove_edge(1,3)
G.remove_edges_from([(1,2),("u","V")])

G.number_of_nodes()
G.number_of_edges()

'''PROGRAM STARTS BENEATH THIS LINE'''
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import numpy as np

#Plot graph with networkx
#Gives warning because networkx is not necessarily used to plot graphs
graph = nx.karate_club_graph()

plt.figure()
nx.draw(graph, with_labels=True, node_color="lightblue", edge_color="gray")
plt.savefig("karate_graph.pdf")

#Check the degree of each nodes, aka how many edges connected to each nodes
#return a dictionary with keys as node IDs
#and the values as their associated degrees.
graph.degree()[33]
graph.degree(33)

#random graphs generator
N = 20
prob = 0.2

def er_graph(N, p):
    '''generate an ER graph.'''
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 < node2 and bernoulli.rvs(p=prob):
                G.add_edge(node1,node2)
    return G

plt.figure()
nx.draw(er_graph(40,0.08), node_size=40, node_color='gray')
plt.savefig("er1.pdf")
       

#Plot Distribution of edges degree
def plot_degree_distribution(G):
    plt.hist(list(G.degree().values()), histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("$P(k)$")
    plt.title("Degree distribution")


plt.figure()
G1 = er_graph(500,0.08)
plot_degree_distribution(G1)
G2 = er_graph(500,0.08)
plot_degree_distribution(G2)
G3 = er_graph(500,0.08)
plot_degree_distribution(G3)
plt.savefig("hist3.pdf")


#VILLAGE NETWORKS IN INDIA#
A1 = np.loadtxt("adj_allVillageRelationships_vilno_1.csv", delimiter=',')
A2 = np.loadtxt("adj_allVillageRelationships_vilno_2.csv", delimiter=',')

G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)


def basic_net_stats(G):
    print("Number of nodes: %d" % G.number_of_nodes())
    print("Number of edges: %d" % G.number_of_edges())
    print("Average degree: %.2f" % np.mean(list(G.degree().values())))


basic_net_stats(G1)
basic_net_stats(G2)

plt.figure()
plot_degree_distribution(G1)
plot_degree_distribution(G2)

plt.savefig("village_hist.pdf")


#Goes over the graph and find the largest components
#LCC for Largest Connected Component

G1_LCC = max(nx.connected_component_subgraphs(G1), key=len)
G2_LCC = max(nx.connected_component_subgraphs(G2), key=len)


G1_LCC.number_of_nodes() / G1.number_of_nodes()
G2_LCC.number_of_nodes() / G2.number_of_nodes()


#Plot LCC for village 1 & 2
plt.figure()
nx.draw(G1_LCC, node_color="red", edge_color="gray", node_size=20)
plt.savefig("village1.pdf")

plt.figure()
nx.draw(G2_LCC, node_color="green", edge_color="gray", node_size=20)
plt.savefig("village2.pdf")
